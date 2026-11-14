import asyncio

from .run_full_execution import main as full_execution_main
from .run_evaluation import main as eval_main
from .run_safety import main as safety_main
from .run_regression import main as regression_main
from .generate_reports import main as reports_main
from .run_multiturn import main as multiturn_main


async def main():
    print('\n=== 1. CHATBOT APPLICATION EXECUTION ===')
    await full_execution_main()
    print('\n=== 2. GOLDEN DATASET EVALUATION ===')
    eval_main()
    print('\n=== 3. SAFETY TESTING ===')
    await safety_main()
    print('\n=== 4. REGRESSION SUITE ===')
    await regression_main()
    print('\n=== 5. MULTI-TURN REGRESSION ===')
    await multiturn_main()
    print('\n=== 6. REPORT GENERATION ===')
    reports_main()
    print('\nComplete. Open reports/chatbot_app/ or run: streamlit run dashboard/app.py')


if __name__ == '__main__':
    asyncio.run(main())
