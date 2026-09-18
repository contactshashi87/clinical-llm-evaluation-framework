import asyncio
import pandas as pd

from src.settings import ROOT
from src.chatbot_executor import ChatbotExecutor


GOLDEN_DATASET = ROOT / "data" / "golden_dataset.csv"
FINAL_OUTPUT = ROOT / "reports" / "generated" / "full_90_execution.csv"


async def main():

    print("=" * 70)
    print("FULL 90-QUESTION EXECUTION")
    print("=" * 70)

    golden_df = pd.read_csv(
    GOLDEN_DATASET,
    dtype=str
).fillna("")

    print(f"Total golden questions: {len(golden_df)}")

    results = []

    executor = ChatbotExecutor()

    for index, row in golden_df.iterrows():

        question_number = index + 1
        question = row["question"]

        print("\n" + "-" * 70)
        print(
            f"[{question_number}/{len(golden_df)}] {question}"
        )
        print("-" * 70)

        try:

            result = await executor.ask(question)

            results.append(
    {
        "question": question,
        "expected_answer": row["expected_answer"],
        "category": row["category"],
        "answer": result.answer,
        "latency_seconds": result.latency_seconds,
        "executor_status": result.status,
        "executor_error": result.error or "",
    }
)

            print(f"Execution status: {result.status}")

        except Exception as exc:

            print(
                f"ERROR: Question {question_number} failed."
            )
            print(
                f"Reason: {type(exc).__name__}: {exc}"
            )

            results.append({
    "question": question,
    "expected_answer": row["expected_answer"],
    "source_reference": row["source_reference"],
    "category": row["category"],
    "answer": "",
    "latency_seconds": "",
    "executor_status": "ERROR",
    "executor_error": f"{type(exc).__name__}: {exc}",
})

    result_df = pd.DataFrame(results)

    FINAL_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        FINAL_OUTPUT,
        index=False
    )

    print("\n" + "=" * 70)
    print("FULL EXECUTION COMPLETE")
    print("=" * 70)

    successful = (
        result_df["executor_status"] == "PASS"
    ).sum()

    print(f"Total processed : {len(result_df)}")
    print(f"Successful      : {successful}")
    print(f"Failed          : {len(result_df) - successful}")

    print("\nSaved:")
    print(f"  {FINAL_OUTPUT}")


if __name__ == "__main__":
    asyncio.run(main())