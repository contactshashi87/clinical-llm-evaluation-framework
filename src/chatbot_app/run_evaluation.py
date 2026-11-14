import ast
import json
from pathlib import Path

import pandas as pd

from .settings import OUTPUT_DIR
from .evaluation_engine import EvaluationEngine
from .reporting import ReportManager


GENERATED_CSV = OUTPUT_DIR / "generated" / "rag_test_set.csv"


def parse_contexts(value):
    """
    Convert the contexts column from the workshop CSV
    into the list[str] format expected by Ragas.
    """
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return [str(x) for x in value]

    text = str(value).strip()

    if not text:
        return []

    # Try Python-list representation
    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, list):
            return [str(x) for x in parsed]

    except (ValueError, SyntaxError):
        pass

    # Try JSON representation
    try:
        parsed = json.loads(text)

        if isinstance(parsed, list):
            return [str(x) for x in parsed]

    except (ValueError, json.JSONDecodeError):
        pass

    # Fallback: treat the value as one context
    return [text]


def main():
    print("Loading generated RAG test dataset...")

    if not GENERATED_CSV.exists():
        raise FileNotFoundError(
            f"Generated test dataset not found:\n{GENERATED_CSV}\n\n"
            "Run the chatbot execution first so that rag_test_set.csv is created."
        )

    df = pd.read_csv(GENERATED_CSV, dtype=str).fillna("")

    required_columns = [
        "question",
        "contexts",
        "answer",
        "ground_truth",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Generated CSV is missing required columns: {missing}"
        )

    print(f"Loaded {len(df)} records.")

    records = []

    for _, row in df.iterrows():
        contexts = parse_contexts(row["contexts"])

        records.append(
            {
                "question": row["question"],
                "answer": row["answer"],
                "expected_answer": row["ground_truth"],
                "contexts": contexts,
                "context_source": "workshop_app",
            }
        )

    usable = [
        record
        for record in records
        if record["question"]
        and record["answer"]
        and record["expected_answer"]
        and record["contexts"]
    ]

    print(f"Usable records for Ragas: {len(usable)}")

    if not usable:
        raise RuntimeError(
            "No usable records found in rag_test_set.csv."
        )

    print("\nRunning Ragas evaluation...")
    print("Metrics:")
    print("  - Faithfulness")
    print("  - Answer Relevance")
    print("  - Context Precision")

    engine = EvaluationEngine()
    evaluated = engine.evaluate(usable)

    summary = engine.summary(evaluated)

    ReportManager().save_evaluation(
        evaluated,
        summary,
    )

    print("\nEvaluation Summary")
    print("==================")
    print(
        pd.DataFrame([summary]).to_string(index=False)
    )

    print("\nSaved:")
    print("  reports/chatbot_app/evaluation_results.csv")
    print("  reports/chatbot_app/evaluation_summary.json")

    return evaluated


if __name__ == "__main__":
    main()