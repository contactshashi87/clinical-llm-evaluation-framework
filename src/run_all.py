from .run_evaluation import main as eval_main
from .run_safety import main as safety_main
from .run_regression import main as regression_main
from .generate_reports import main as reports_main
from .run_multiturn import main as multiturn_main

def main():
    print('\n=== 1. GOLDEN DATASET EVALUATION ===')
    eval_main()
    print('\n=== 2. SAFETY TESTING ===')
    safety_main()
    print('\n=== 3. REGRESSION SUITE ===')
    regression_main()
    print('\n=== 4. MULTI-TURN REGRESSION ===')
    multiturn_main()
    print('\n=== 5. REPORT GENERATION ===')
    reports_main()
    print('\nComplete. Open reports/ and run: streamlit run dashboard/app.py')
if __name__=='__main__': main()
