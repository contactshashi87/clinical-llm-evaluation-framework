import os, math
from pathlib import Path
import pandas as pd
from src.settings import EVALUATION
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from sklearn.metrics.pairwise import cosine_similarity
from .settings import EVAL_MODEL, EMBED_MODEL, THRESHOLDS, OUTPUT_DIR

class EvaluationEngine:
    def __init__(self):
        key = os.getenv('OPENAI_API_KEY')
        if not key: raise ValueError('OPENAI_API_KEY is not set. Put it in .env or the environment.')
        self.llm = LangchainLLMWrapper(ChatOpenAI(model=EVAL_MODEL, temperature=0, api_key=key))
        self.embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBED_MODEL, api_key=key))
        self.raw_embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=key)

    def evaluate(self, records):
        usable = [
            r for r in records
            if r.get('answer')
            and r.get('contexts')
        ]

        if not usable:
            raise RuntimeError(
                'No usable records. Chatbot execution produced no answers/contexts.'
            )

        # ---------------------------------------------------------
        # Skip Ragas when evaluation is disabled
        # ---------------------------------------------------------
        if not EVALUATION.get("enabled", True):
            print("\nRagas evaluation is disabled.")
            print("Evaluation metrics will be recorded as NOT_EVALUATED.")

            base = pd.DataFrame(usable).reset_index(drop=True)

            base["faithfulness"] = float("nan")
            base["answer_relevancy"] = float("nan")
            base["context_precision"] = float("nan")
            base["similarity_score"] = float("nan")
            base["hallucination_flag"] = float("nan")
            base["status"] = "NOT_EVALUATED"

            return base

        # ---------------------------------------------------------
        # Create Ragas dataset
        # ---------------------------------------------------------
        ds = Dataset.from_list([
            {
                'user_input': r['question'],
                'retrieved_contexts': r['contexts'],
                'response': r['answer'],
                'reference': r['expected_answer']
            }
            for r in usable
        ])

        base = pd.DataFrame(usable).reset_index(drop=True)

        # Default evaluation columns.
        # NaN means the metric was not evaluated.
        base['faithfulness'] = float('nan')
        base['answer_relevancy'] = float('nan')
        base['context_precision'] = float('nan')

        try:
            print("\nRunning Ragas evaluation...")

            result = evaluate(
                ds,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision
                ],
                llm=self.llm,
                embeddings=self.embeddings
            )

            df = result.to_pandas()

            for col in [
                'faithfulness',
                'answer_relevancy',
                'context_precision'
            ]:
                if col in df.columns:
                    base[col] = pd.to_numeric(
                        df[col],
                        errors='coerce'
                    )

        except Exception as exc:
            error_text = str(exc).lower()

            if (
                "credit_balance_exhausted" in error_text
                or "no credits remaining" in error_text
                or "insufficient_quota" in error_text
                or "rate limit" in error_text
                or "429" in error_text
            ):
                print("\nWARNING: OpenAI API credits unavailable.")
            else:
                print("\nWARNING: Ragas evaluation could not be completed.")
                print(f"Reason: {type(exc).__name__}: {exc}")

            print(
                "Evaluation metrics will be recorded as NaN "
                "and the framework will continue."
            )

        # ---------------------------------------------------------
        # Optional similarity score
        # ---------------------------------------------------------
        base['similarity_score'] = float('nan')

        # ---------------------------------------------------------
        # Hallucination flag
        # ---------------------------------------------------------
        base['hallucination_flag'] = base['faithfulness'].apply(
            lambda x: (
                float('nan')
                if pd.isna(x)
                else x < 0.70
            )
        )

        # ---------------------------------------------------------
        # Final status
        # ---------------------------------------------------------
        base['status'] = base.apply(
            self._status,
            axis=1
        )

        return base

    def _similarity(self, df):
        refs = df['expected_answer'].tolist(); answers = df['answer'].tolist()
        ref_vec = self.raw_embeddings.embed_documents(refs)
        ans_vec = self.raw_embeddings.embed_documents(answers)
        return [float(cosine_similarity([a],[b])[0][0]) for a,b in zip(ans_vec, ref_vec)]

    @staticmethod
    def _status(r):
        vals = [
            r['answer_relevancy'],
            r['faithfulness'],
            r['context_precision']
        ]

        thresholds = [
            THRESHOLDS['relevance'],
            THRESHOLDS['faithfulness'],
            THRESHOLDS['precision']
        ]

        if any(pd.isna(v) for v in vals):
            return 'NOT_EVALUATED'

        return (
            'PASS'
            if all(v >= t for v, t in zip(vals, thresholds))
            else 'FAIL'
        )

    @staticmethod
    def summary(df):
        evaluated = df[df['status'].isin(['PASS', 'FAIL'])]

        pass_rate = (
            (evaluated['status'] == 'PASS').mean() * 100
            if len(evaluated) > 0
            else float('nan')
        )

        hallucination_rate = (
            df['hallucination_flag'].dropna().mean() * 100
            if df['hallucination_flag'].notna().any()
            else float('nan')
        )

        return {
            'total': int(len(df)),
            'passed': int((df.status == 'PASS').sum()),
            'failed': int((df.status == 'FAIL').sum()),
            'pass_rate': round(float(pass_rate), 2)
                if not pd.isna(pass_rate) else None,

            'avg_relevance': round(float(df.answer_relevancy.mean()), 4)
                if not pd.isna(df.answer_relevancy.mean()) else None,

            'avg_faithfulness': round(float(df.faithfulness.mean()), 4)
                if not pd.isna(df.faithfulness.mean()) else None,

            'avg_precision': round(float(df.context_precision.mean()), 4)
                if not pd.isna(df.context_precision.mean()) else None,

            'avg_similarity': round(float(df.similarity_score.mean()), 4)
                if not pd.isna(df.similarity_score.mean()) else None,

            'hallucination_rate': round(float(hallucination_rate), 2)
                if not pd.isna(hallucination_rate) else None,
        }
