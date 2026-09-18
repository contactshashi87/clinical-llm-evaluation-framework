import asyncio
import pandas as pd

from .settings import ROOT
from .chatbot_executor import ChatbotExecutor
from .retriever import LocalRetriever
from .evaluation_engine import EvaluationEngine
from .reporting import ReportManager


async def main():

    print("======================================")
    print("Regression Test Suite")
    print("======================================")

    data = pd.read_csv(
        ROOT / "data/regression_suite.csv"
    )

    print(f"Total regression cases: {len(data)}")
    print()

    retriever = LocalRetriever()
    records = []

    executor = ChatbotExecutor()

    try:

        await executor._ensure_page()

        for index, row in data.iterrows():

            case_id = str(row["id"])
            scenario = str(row["scenario"])

            print(
                f"[{index + 1}/{len(data)}] "
                f"Running {case_id}: {scenario}"
            )

            try:

                result = await executor.ask_safety(
                    scenario
                )

                contexts = (
                    result.contexts
                    if result.contexts
                    else retriever.retrieve(scenario)
                )

                records.append(
    {
        **row.to_dict(),
        "question": scenario,
        "answer": result.answer,
        "contexts": contexts,
        "latency_seconds": result.latency_seconds,
        "executor_status": result.status,
    }
)

                print(
                    f"    Executor status: {result.status}"
                )

            except Exception as exc:

                print(
                    f"    ERROR: {exc}"
                )

                records.append(
    {
        **row.to_dict(),
        "question": scenario,
        "answer": "",
        "contexts": [],
        "latency_seconds": float("nan"),
        "executor_status": "ERROR",
    }
)

        print()
        print("======================================")
        print("Running LLM Evaluation")
        print("======================================")
        print(
            "Evaluation is enabled. "
            "Attempting Relevance, Faithfulness "
            "and Context Precision..."
        )
        print()

        evaluated = EvaluationEngine().evaluate(
            records
        )

        # Count execution status
        total = len(evaluated)

        passed = int(
            (evaluated["status"] == "PASS").sum()
        )

        failed = int(
            (evaluated["status"] == "FAIL").sum()
        )

        # Metrics may legitimately be NaN when
        # the LLM provider has no available credits.
        avg_relevance = (
            evaluated["answer_relevancy"].mean()
        )

        avg_faithfulness = (
            evaluated["faithfulness"].mean()
        )

        avg_precision = (
            evaluated["context_precision"].mean()
        )

        if pd.isna(avg_relevance):
            relevance_report = "NAN - NO CREDIT"
        else:
            relevance_report = avg_relevance

        if pd.isna(avg_faithfulness):
            faithfulness_report = "NAN - NO CREDIT"
        else:
            faithfulness_report = avg_faithfulness

        if pd.isna(avg_precision):
            precision_report = "NAN - NO CREDIT"
        else:
            precision_report = avg_precision

        summary = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": (
                float((passed / total) * 100)
                if total
                else None
            ),
            "avg_relevance": relevance_report,
            "avg_faithfulness": faithfulness_report,
            "avg_precision": precision_report,
            "evaluation_status": (
                "NAN - NO CREDIT"
                if (
                    pd.isna(avg_relevance)
                    and pd.isna(avg_faithfulness)
                    and pd.isna(avg_precision)
                )
                else "EVALUATED"
            ),
        }

        ReportManager().save_regression(
            evaluated,
            summary
        )

        print()
        print("======================================")
        print("Regression Results")
        print("======================================")

        print(
            evaluated[
                [
                    "id",
                    "status",
                    "answer_relevancy",
                    "faithfulness",
                    "context_precision",
                ]
            ].to_string(index=False)
        )

        print()
        print("======================================")
        print("Regression Summary")
        print("======================================")

        print(f"Total cases       : {total}")
        print(f"Passed            : {passed}")
        print(f"Failed            : {failed}")
        print(
            f"Pass rate         : "
            f"{summary['pass_rate']}"
        )
        print(
            f"Avg Relevance     : "
            f"{relevance_report}"
        )
        print(
            f"Avg Faithfulness  : "
            f"{faithfulness_report}"
        )
        print(
            f"Avg Precision     : "
            f"{precision_report}"
        )
        print(
            f"Evaluation status : "
            f"{summary['evaluation_status']}"
        )

        print()
        print(
            "Saved: reports/regression_results.csv"
        )

        return evaluated

    finally:

        await executor.close()


if __name__ == "__main__":
    asyncio.run(main())