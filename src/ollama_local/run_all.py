import asyncio

from .final_benchmarking_report import main as report_main
from .ollama_rag_evaluation import main as rag_main
from .ollama_safety_evaluation import main as safety_main


def main():
    print("\n=== OLLAMA LOCAL RAG EVALUATION ===")
    rag_main()
    print("\n=== OLLAMA LOCAL SAFETY EVALUATION ===")
    asyncio.run(safety_main())
    print("\n=== OLLAMA LOCAL BENCHMARK REPORT ===")
    report_main()


if __name__ == "__main__":
    main()