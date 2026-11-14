import asyncio
import json
import math

import ollama
import pandas as pd
from datasets import Dataset
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.outputs import LLMResult, Generation

from ragas import evaluate
from ragas.llms import BaseRagasLLM
from ragas.embeddings import BaseRagasEmbeddings
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig

from ..shared.paths import ROOT
from ..shared.retriever import LocalRetriever
from .settings import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    OLLAMA_HOST,
    QUESTIONS_TO_EVALUATE,
)


# ============================================================
# CONFIGURATION
# ============================================================

GOLDEN_DATASET = ROOT / "data" / "shared" / "golden_dataset.csv"
REPORTS_DIR = ROOT / "reports" / "ollama_local"

RESULTS_CSV = REPORTS_DIR / "ollama_ragas_results.csv"
SUMMARY_JSON = REPORTS_DIR / "ollama_ragas_summary.json"
REPORT_PDF = REPORTS_DIR / "Ollama_RAG_Evaluation_Report.pdf"

# Required thresholds from the assignment
RELEVANCE_THRESHOLD = 0.80
FAITHFULNESS_THRESHOLD = 0.85
PRECISION_THRESHOLD = 0.80

# Local execution: one worker avoids overloading the machine.
RUN_CONFIG = RunConfig(
    timeout=180,
    max_retries=1,
    max_workers=1,
)


# ============================================================
# OLLAMA RAGAS LLM ADAPTER
# ============================================================

class OllamaRagasLLM(BaseRagasLLM):

    def __init__(self, model, host):
        self.model = model
        self.client = ollama.Client(host=host)

    @staticmethod
    def _extract_schema(prompt_text):
        """
        Extract the JSON schema that Ragas places in its prompt.

        Ragas 0.3.9 may include a JSON Schema section in the
        evaluation prompt. Passing that exact schema to Ollama
        is more reliable than using generic format="json".
        """
        markers = [
            "JSON Schema:",
            "JSON schema:",
            "json schema:",
        ]

        start = -1

        for marker in markers:
            position = prompt_text.find(marker)
            if position != -1:
                start = position + len(marker)
                break

        if start == -1:
            return None

        schema_text = prompt_text[start:].strip()
        decoder = json.JSONDecoder()

        try:
            schema, _ = decoder.raw_decode(schema_text)
            if isinstance(schema, dict):
                return schema
        except Exception:
            pass

        # The prompt can contain text before the actual JSON object.
        # Try every opening brace until a valid JSON object is found.
        for index, character in enumerate(schema_text):
            if character != "{":
                continue

            try:
                schema, _ = decoder.raw_decode(schema_text[index:])
                if isinstance(schema, dict):
                    return schema
            except Exception:
                continue

        return None

    @staticmethod
    def _clean_json_text(text):
        """Remove accidental markdown fences around JSON."""
        if not text:
            return ""

        text = text.strip()

        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        return text.strip()

    def generate_text(
        self,
        prompt,
        n=1,
        temperature=0.01,
        stop=None,
        callbacks=None
    ):
        """
        Generate one or more Ragas evaluator responses.

        Ragas expects the LLM adapter to return LangChain
        Generation objects. When Ragas provides an output JSON
        schema, that schema is passed directly to Ollama.
        """
        prompt_text = prompt.to_string()
        schema = self._extract_schema(prompt_text)

        generations = []

        for _ in range(n):
            response_kwargs = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a strict evaluation engine. "
                            "Follow the evaluation instructions exactly. "
                            "Return ONLY the requested JSON object. "
                            "Do not provide explanations outside JSON. "
                            "Do not use markdown or code fences."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt_text
                    }
                ],
                "options": {
                    "temperature": 0
                }
            }

            # Prefer Ragas' exact schema when available.
            if schema is not None:
                response_kwargs["format"] = schema
            else:
                response_kwargs["format"] = "json"

            response = self.client.chat(**response_kwargs)

            if isinstance(response, dict):
                message = response.get("message", {})
                text = message.get("content", "")
            else:
                text = response.message.content

            text = self._clean_json_text(text)

            # If Ollama still returned something that is not JSON,
            # retry once with an explicit JSON-only instruction.
            try:
                json.loads(text)
            except Exception:
                retry_messages = [
                    {
                        "role": "system",
                        "content": (
                            "Return ONLY valid JSON. "
                            "Follow the requested evaluation schema exactly. "
                            "No explanation. No markdown. No code fences."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt_text
                    }
                ]

                retry_kwargs = {
                    "model": self.model,
                    "messages": retry_messages,
                    "options": {
                        "temperature": 0
                    }
                }

                if schema is not None:
                    retry_kwargs["format"] = schema
                else:
                    retry_kwargs["format"] = "json"

                retry_response = self.client.chat(**retry_kwargs)

                if isinstance(retry_response, dict):
                    retry_message = retry_response.get(
                        "message",
                        {}
                    )
                    text = retry_message.get(
                        "content",
                        ""
                    )
                else:
                    text = retry_response.message.content

                text = self._clean_json_text(text)

            generations.append(
                Generation(text=text)
            )

        return LLMResult(
            generations=[generations]
        )

    async def agenerate_text(
        self,
        prompt,
        n=1,
        temperature=0.01,
        stop=None,
        callbacks=None
    ):
        return await asyncio.to_thread(
            self.generate_text,
            prompt,
            n,
            temperature,
            stop,
            callbacks
        )

    def is_finished(self, response):
        return True


# ============================================================
# OLLAMA EMBEDDING ADAPTER
# ============================================================

class OllamaRagasEmbeddings(BaseRagasEmbeddings):

    def __init__(self, model, host):
        self.model = model
        self.client = ollama.Client(host=host)

    def _get_embeddings(self, response):

        if isinstance(response, dict):
            return response.get("embeddings", [])

        return response.embeddings

    def embed_documents(self, texts):

        if not texts:
            return []

        response = self.client.embed(
            model=self.model,
            input=texts,
        )

        return self._get_embeddings(response)

    def embed_query(self, text):

        response = self.client.embed(
            model=self.model,
            input=text,
        )

        embeddings = self._get_embeddings(response)

        if not embeddings:
            raise RuntimeError(
                "Ollama returned no embedding."
            )

        if isinstance(embeddings[0], list):
            return embeddings[0]

        return embeddings

    async def aembed_documents(self, texts):

        return await asyncio.to_thread(
            self.embed_documents,
            texts,
        )

    async def aembed_query(self, text):

        return await asyncio.to_thread(
            self.embed_query,
            text,
        )


# ============================================================
# LOAD 10 GOLDEN QUESTIONS
# ============================================================

def load_questions():

    if not GOLDEN_DATASET.exists():
        raise FileNotFoundError(
            f"Golden dataset not found: {GOLDEN_DATASET}"
        )

    df = pd.read_csv(
        GOLDEN_DATASET
    )

    # Actual schema of your golden_dataset.csv:
    #
    # id
    # question
    # expected_answer
    # source
    # category
    # risk_level

    required_columns = [
        "id",
        "question",
        "expected_answer",
        "source",
        "category",
        "risk_level",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Golden dataset missing columns: {missing}"
        )

    if len(df) < QUESTIONS_TO_EVALUATE:
        raise RuntimeError(
            f"Golden dataset contains only {len(df)} "
            f"questions. Need at least "
            f"{QUESTIONS_TO_EVALUATE}."
        )

    # Use the first 10 questions deterministically.
    selected = df.head(
        QUESTIONS_TO_EVALUATE
    ).copy()

    return selected


# ============================================================
# GENERATE ANSWER USING LOCAL OLLAMA
# ============================================================

def generate_answer(
    client,
    question,
    contexts,
):

    context_text = "\n\n---\n\n".join(
        contexts
    )

    prompt = f"""
You are a Clinical Review Assistant being evaluated.

Answer the user's question using ONLY the provided
knowledge-base context.

Rules:
1. Do not invent facts.
2. Do not use outside medical knowledge.
3. Do not make unsupported clinical recommendations.
4. If the context does not contain enough information,
   clearly state that the available knowledge base does
   not contain enough information.
5. Give a concise answer.

User Question:
{question}

Knowledge Base Context:
{context_text}

Answer:
"""

    response = client.generate(
        model=LLM_MODEL,
        prompt=prompt,
        options={
            "temperature": 0.0,
        },
    )

    if isinstance(response, dict):
        answer = response.get(
            "response",
            "",
        )
    else:
        answer = response.response

    return answer.strip()


# ============================================================
# SIMILARITY SCORE
# ============================================================

def calculate_similarity(
    embeddings,
    expected_answers,
    generated_answers,
):

    reference_vectors = (
        embeddings.embed_documents(
            expected_answers
        )
    )

    answer_vectors = (
        embeddings.embed_documents(
            generated_answers
        )
    )

    scores = []

    for reference_vector, answer_vector in zip(
        reference_vectors,
        answer_vectors,
    ):

        score = cosine_similarity(
            [answer_vector],
            [reference_vector],
        )[0][0]

        scores.append(
            float(score)
        )

    return scores


# ============================================================
# HALLUCINATION FLAG
# ============================================================

def hallucination_flag(
    faithfulness_score
):

    if pd.isna(
        faithfulness_score
    ):
        return float("nan")

    return bool(
        faithfulness_score < 0.70
    )


# ============================================================
# STATUS
# ============================================================

def calculate_status(row):

    relevance = row[
        "answer_relevancy"
    ]

    faithfulness_score = row[
        "faithfulness"
    ]

    precision = row[
        "context_precision"
    ]

    if any(
        pd.isna(value)
        for value in [
            relevance,
            faithfulness_score,
            precision,
        ]
    ):
        return "NOT_EVALUATED"

    if (
        relevance >= RELEVANCE_THRESHOLD
        and
        faithfulness_score >= FAITHFULNESS_THRESHOLD
        and
        precision >= PRECISION_THRESHOLD
    ):
        return "PASS"

    return "FAIL"


# ============================================================
# SAFE NUMBER
# ============================================================

def safe_number(value):

    try:

        value = float(value)

        if math.isnan(value):
            return None

        return value

    except Exception:
        return None


# ============================================================
# PDF REPORT
# ============================================================

def generate_pdf_report(
    results_df,
    summary,
):

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import (
        getSampleStyleSheet
    )
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    document = SimpleDocTemplate(
        str(REPORT_PDF),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    story = []

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Ollama RAG Evaluation Report",
            styles["Title"],
        )
    )

    story.append(
        Spacer(1, 8)
    )

    story.append(
        Paragraph(
            "Clinical Review Assistant",
            styles["Heading2"],
        )
    )

    story.append(
        Spacer(1, 18)
    )

    # --------------------------------------------------------
    # EVALUATION CONFIGURATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Evaluation",
            styles["Heading2"],
        )
    )

    config_data = [
        [
            "Evaluation Model",
            LLM_MODEL,
        ],
        [
            "Embedding Model",
            EMBEDDING_MODEL,
        ],
        [
            "Questions Evaluated",
            str(
                summary[
                    "questions_evaluated"
                ]
            ),
        ],
        [
            "Ragas Version",
            "0.3.9",
        ],
        [
            "Knowledge Retrieval",
            "Local TF-IDF Retriever",
        ],
    ]

    config_table = Table(
        config_data,
        colWidths=[
            2.0 * inch,
            5.0 * inch,
        ],
    )

    config_table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.lightgrey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
            ]
        )
    )

    story.append(
        config_table
    )

    story.append(
        Spacer(1, 18)
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Evaluation Results",
            styles["Heading2"],
        )
    )

    def metric(value):
        value = safe_number(value)

        if value is None:
            return "N/A"

        return f"{value:.4f}"

    def percentage(value):
        value = safe_number(value)

        if value is None:
            return "N/A"

        return f"{value:.2f}%"

    metrics_data = [
        [
            "Metric",
            "Average",
            "Type",
        ],
        [
            "Answer Relevance",
            metric(
                summary[
                    "avg_relevance"
                ]
            ),
            "Required",
        ],
        [
            "Faithfulness",
            metric(
                summary[
                    "avg_faithfulness"
                ]
            ),
            "Required",
        ],
        [
            "Context Precision",
            metric(
                summary[
                    "avg_precision"
                ]
            ),
            "Required",
        ],
        [
            "Context Recall",
            metric(
                summary[
                    "avg_context_recall"
                ]
            ),
            "Optional",
        ],
        [
            "Similarity Score",
            metric(
                summary[
                    "avg_similarity"
                ]
            ),
            "Optional",
        ],
        [
            "Hallucination Rate",
            percentage(
                summary[
                    "hallucination_rate"
                ]
            ),
            "Optional",
        ],
    ]

    metrics_table = Table(
        metrics_data,
        colWidths=[
            3.0 * inch,
            2.0 * inch,
            1.5 * inch,
        ],
    )

    metrics_table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "ALIGN",
                    (1, 1),
                    (-1, -1),
                    "CENTER",
                ),
            ]
        )
    )

    story.append(
        metrics_table
    )

    story.append(
        Spacer(1, 18)
    )

    # --------------------------------------------------------
    # PASS / FAIL
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Acceptance Results",
            styles["Heading2"],
        )
    )

    acceptance_data = [
        [
            "Threshold",
            "Required Value",
        ],
        [
            "Answer Relevance",
            f">= {RELEVANCE_THRESHOLD}",
        ],
        [
            "Faithfulness",
            f">= {FAITHFULNESS_THRESHOLD}",
        ],
        [
            "Context Precision",
            f">= {PRECISION_THRESHOLD}",
        ],
        [
            "Questions Passed",
            str(
                summary["passed"]
            ),
        ],
        [
            "Questions Failed",
            str(
                summary["failed"]
            ),
        ],
        [
            "Pass Rate",
            percentage(
                summary["pass_rate"]
            ),
        ],
    ]

    acceptance_table = Table(
        acceptance_data,
        colWidths=[
            3.0 * inch,
            3.5 * inch,
        ],
    )

    acceptance_table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
            ]
        )
    )

    story.append(
        acceptance_table
    )

    story.append(
        Spacer(1, 18)
    )

    # --------------------------------------------------------
    # PER QUESTION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Per-Question Results",
            styles["Heading2"],
        )
    )

    detail_data = [
        [
            "ID",
            "Category",
            "Relevance",
            "Faith.",
            "Precision",
            "Recall",
            "Status",
        ]
    ]

    for _, row in results_df.iterrows():

        detail_data.append(
            [
                str(
                    row["question_id"]
                ),
                str(
                    row["category"]
                )[:18],
                metric(
                    row[
                        "answer_relevancy"
                    ]
                ),
                metric(
                    row[
                        "faithfulness"
                    ]
                ),
                metric(
                    row[
                        "context_precision"
                    ]
                ),
                metric(
                    row[
                        "context_recall"
                    ]
                ),
                str(
                    row["status"]
                ),
            ]
        )

    detail_table = Table(
        detail_data,
        repeatRows=1,
        colWidths=[
            0.55 * inch,
            1.55 * inch,
            0.8 * inch,
            0.75 * inch,
            0.8 * inch,
            0.7 * inch,
            0.85 * inch,
        ],
    )

    detail_table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.grey,
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "ALIGN",
                    (2, 1),
                    (-1, -1),
                    "CENTER",
                ),
            ]
        )
    )

    story.append(
        detail_table
    )

    story.append(
        Spacer(1, 18)
    )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "This report represents an automated LLM evaluation "
            "framework result. It is not clinical validation and "
            "does not replace clinical governance or expert review.",
            styles["BodyText"],
        )
    )

    document.build(
        story
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("OLLAMA RAG EVALUATION")
    print("=" * 60)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # CHECK OLLAMA
    # --------------------------------------------------------

    print()
    print("Checking Ollama...")

    client = ollama.Client(
        host=OLLAMA_HOST
    )

    try:

        models_response = client.list()

        if isinstance(
            models_response,
            dict,
        ):
            models = models_response.get(
                "models",
                [],
            )
        else:
            models = models_response.models

        model_names = []

        for model in models:

            if isinstance(
                model,
                dict,
            ):
                model_names.append(
                    model.get(
                        "name",
                        "",
                    )
                )
            else:
                model_names.append(
                    model.model
                )

        if not any(
            name.startswith(
                LLM_MODEL
            )
            for name in model_names
        ):
            raise RuntimeError(
                f"Model '{LLM_MODEL}' "
                "is not installed."
            )

        if not any(
            name.startswith(
                EMBEDDING_MODEL
            )
            for name in model_names
        ):
            raise RuntimeError(
                f"Embedding model "
                f"'{EMBEDDING_MODEL}' "
                "is not installed."
            )

    except Exception as exc:

        print()
        print(
            "ERROR: Could not connect to Ollama."
        )
        print(
            f"Reason: {type(exc).__name__}: {exc}"
        )
        return

    print(
        "Ollama connection : OK"
    )

    print(
        f"LLM model         : {LLM_MODEL}"
    )

    print(
        f"Embedding model   : {EMBEDDING_MODEL}"
    )

    # --------------------------------------------------------
    # LOAD QUESTIONS
    # --------------------------------------------------------

    questions_df = load_questions()

    print()
    print(
        f"Questions evaluated : "
        f"{len(questions_df)}"
    )

    # --------------------------------------------------------
    # RETRIEVER
    # --------------------------------------------------------

    retriever = LocalRetriever(
        top_k=4
    )

    # --------------------------------------------------------
    # GENERATE ANSWERS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("GENERATING OLLAMA ANSWERS")
    print("=" * 60)

    records = []

    for index, row in questions_df.iterrows():

        number = index + 1

        question_id = str(
            row["id"]
        )

        question = str(
            row["question"]
        )

        expected_answer = str(
            row["expected_answer"]
        )

        source = str(
            row["source"]
        )

        category = str(
            row["category"]
        )

        risk_level = str(
            row["risk_level"]
        )

        print(
            f"[{number}/{len(questions_df)}] "
            f"{question}"
        )

        contexts = retriever.retrieve(
            question
        )

        try:

            answer = generate_answer(
                client,
                question,
                contexts,
            )

            print(
                f"  Answer generated."
            )

        except Exception as exc:

            print(
                f"  ERROR: "
                f"{type(exc).__name__}: {exc}"
            )

            answer = ""

        records.append(
            {
                "question_id": question_id,
                "question": question,
                "expected_answer": expected_answer,
                "source_reference": source,
                "category": category,
                "risk_level": risk_level,
                "contexts": contexts,
                "answer": answer,
            }
        )

    usable_records = [
        record
        for record in records
        if record["answer"]
        and record["contexts"]
    ]

    if not usable_records:

        print()
        print(
            "ERROR: No usable records."
        )
        return

    # --------------------------------------------------------
    # CREATE RAGAS DATASET
    # --------------------------------------------------------

    dataset = Dataset.from_list(
        [
            {
                "user_input": record[
                    "question"
                ],
                "retrieved_contexts": record[
                    "contexts"
                ],
                "response": record[
                    "answer"
                ],
                "reference": record[
                    "expected_answer"
                ],
            }
            for record in usable_records
        ]
    )

    # --------------------------------------------------------
    # RAGAS COMPONENTS
    # --------------------------------------------------------

    evaluator_llm = OllamaRagasLLM(
        model=LLM_MODEL,
        host=OLLAMA_HOST,
    )

    evaluator_embeddings = (
        OllamaRagasEmbeddings(
            model=EMBEDDING_MODEL,
            host=OLLAMA_HOST,
        )
    )

    # --------------------------------------------------------
    # RUN RAGAS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RUNNING RAGAS EVALUATION")
    print("=" * 60)

    results_df = pd.DataFrame(
        usable_records
    )

    results_df[
        "answer_relevancy"
    ] = float("nan")

    results_df[
        "faithfulness"
    ] = float("nan")

    results_df[
        "context_precision"
    ] = float("nan")

    results_df[
        "context_recall"
    ] = float("nan")

    try:

        ragas_result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=RUN_CONFIG,
            raise_exceptions=False,
        )

        ragas_df = (
            ragas_result.to_pandas()
        )

        for column in [
            "answer_relevancy",
            "faithfulness",
            "context_precision",
            "context_recall",
        ]:

            if column in ragas_df.columns:

                results_df[
                    column
                ] = pd.to_numeric(
                    ragas_df[
                        column
                    ],
                    errors="coerce",
                ).values

    except Exception as exc:

        print()
        print(
            "WARNING: Ragas evaluation failed."
        )

        print(
            f"Reason: {type(exc).__name__}: {exc}"
        )

        print(
            "Ragas metrics will remain NAN."
        )

    # --------------------------------------------------------
    # SIMILARITY SCORE
    # --------------------------------------------------------

    print()
    print(
        "Calculating similarity scores..."
    )

    results_df[
        "similarity_score"
    ] = float("nan")

    try:

        similarity_scores = (
            calculate_similarity(
                evaluator_embeddings,
                results_df[
                    "expected_answer"
                ].tolist(),
                results_df[
                    "answer"
                ].tolist(),
            )
        )

        results_df[
            "similarity_score"
        ] = similarity_scores

    except Exception as exc:

        print(
            "WARNING: Similarity calculation failed."
        )

        print(
            f"Reason: {type(exc).__name__}: {exc}"
        )

    # --------------------------------------------------------
    # HALLUCINATION RATE
    # --------------------------------------------------------

    results_df[
        "hallucination_flag"
    ] = results_df[
        "faithfulness"
    ].apply(
        hallucination_flag
    )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    results_df[
        "status"
    ] = results_df.apply(
        calculate_status,
        axis=1,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    evaluated_df = results_df[
        results_df["status"].isin(
            [
                "PASS",
                "FAIL",
            ]
        )
    ]

    evaluated_count = len(
        evaluated_df
    )

    passed_count = int(
        (
            results_df["status"]
            == "PASS"
        ).sum()
    )

    failed_count = int(
        (
            results_df["status"]
            == "FAIL"
        ).sum()
    )

    if evaluated_count:

        pass_rate = (
            passed_count
            / evaluated_count
            * 100
        )

    else:

        pass_rate = float("nan")

    def average(column):

        value = results_df[
            column
        ].mean()

        if pd.isna(value):
            return float("nan")

        return float(value)

    hallucination_values = (
        results_df[
            "hallucination_flag"
        ].dropna()
    )

    if len(
        hallucination_values
    ):

        hallucination_rate = (
            hallucination_values.mean()
            * 100
        )

    else:

        hallucination_rate = float(
            "nan"
        )

    summary = {
        "evaluation_mode": "Local Ollama",
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "ragas_version": "0.3.9",
        "questions_evaluated": len(
            results_df
        ),
        "evaluated": evaluated_count,
        "passed": passed_count,
        "failed": failed_count,
        "pass_rate": pass_rate,
        "avg_relevance": average(
            "answer_relevancy"
        ),
        "avg_faithfulness": average(
            "faithfulness"
        ),
        "avg_precision": average(
            "context_precision"
        ),
        "avg_context_recall": average(
            "context_recall"
        ),
        "avg_similarity": average(
            "similarity_score"
        ),
        "hallucination_rate": (
            hallucination_rate
        ),
        "status": (
            "EVALUATED"
            if evaluated_count
            else "NOT_EVALUATED"
        ),
    }

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    output_columns = [
        "question_id",
        "question",
        "expected_answer",
        "source_reference",
        "category",
        "risk_level",
        "answer",
        "contexts",
        "answer_relevancy",
        "faithfulness",
        "context_precision",
        "context_recall",
        "similarity_score",
        "hallucination_flag",
        "status",
    ]

    results_df[
        output_columns
    ].to_csv(
        RESULTS_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    json_summary = {}

    for key, value in summary.items():

        if isinstance(
            value,
            float,
        ) and math.isnan(value):

            json_summary[key] = None

        else:

            json_summary[key] = value

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_summary,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # CONSOLE RESULTS
    # --------------------------------------------------------

    def display_metric(value):

        value = safe_number(
            value
        )

        if value is None:
            return "NAN"

        return f"{value:.4f}"

    def display_percentage(value):

        value = safe_number(
            value
        )

        if value is None:
            return "NAN"

        return f"{value:.2f}%"

    print()
    print("=" * 60)
    print("RAGAS RESULTS")
    print("=" * 60)

    print()
    print(
        f"Avg Relevance         : "
        f"{display_metric(summary['avg_relevance'])}"
    )

    print(
        f"Avg Faithfulness      : "
        f"{display_metric(summary['avg_faithfulness'])}"
    )

    print(
        f"Avg Context Precision : "
        f"{display_metric(summary['avg_precision'])}"
    )

    print(
        f"Avg Context Recall    : "
        f"{display_metric(summary['avg_context_recall'])}"
    )

    print()
    print("Optional Metrics")

    print(
        f"Avg Similarity        : "
        f"{display_metric(summary['avg_similarity'])}"
    )

    print(
        f"Hallucination Rate    : "
        f"{display_percentage(summary['hallucination_rate'])}"
    )

    print()
    print(
        f"Passed                : "
        f"{passed_count}"
    )

    print(
        f"Failed                : "
        f"{failed_count}"
    )

    print(
        f"Not Evaluated         : "
        f"{len(results_df) - evaluated_count}"
    )

    print(
        f"Pass Rate             : "
        f"{display_percentage(pass_rate)}"
    )

    # --------------------------------------------------------
    # GENERATE PDF
    # --------------------------------------------------------

    print()
    print(
        "Generating PDF report..."
    )

    try:

        generate_pdf_report(
            results_df,
            summary,
        )

    except Exception as exc:

        print(
            "WARNING: PDF generation failed."
        )

        print(
            f"Reason: {type(exc).__name__}: {exc}"
        )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("SAVED")
    print("=" * 60)

    print()
    print(
        f"CSV     : {RESULTS_CSV}"
    )

    print(
        f"Summary : {SUMMARY_JSON}"
    )

    print(
        f"PDF     : {REPORT_PDF}"
    )

    print()
    print("=" * 60)
    print("OLLAMA RAG EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()