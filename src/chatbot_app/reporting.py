from pathlib import Path
import json, pandas as pd
from .settings import ROOT, OUTPUT_DIR, THRESHOLDS

class ReportManager:
    def __init__(self):
        (OUTPUT_DIR/'raw').mkdir(parents=True, exist_ok=True)

    def save_evaluation(self, df, summary):
        df.to_csv(OUTPUT_DIR/'evaluation_results.csv', index=False)
        with open(OUTPUT_DIR/'evaluation_summary.json','w',encoding='utf-8') as f: json.dump(summary,f,indent=2)

    def save_safety(self, df): df.to_csv(OUTPUT_DIR/'safety_results.csv', index=False)
    def save_regression(self, df, summary):
        df.to_csv(OUTPUT_DIR/'regression_results.csv', index=False)
        with open(OUTPUT_DIR/'regression_summary.json','w',encoding='utf-8') as f: json.dump(summary,f,indent=2)

    def generate_benchmark(self, eval_df=None, safety_df=None, regression_df=None):
        lines=['# Final Benchmarking Report','', '## Executive Summary', '', 'Clinical Review Assistant LLM Evaluation Framework execution summary.']
        if eval_df is not None and len(eval_df):
            lines += ['', '## Evaluation Results', '', f"- Dataset executed: {len(eval_df)}", f"- Average Relevance: {eval_df.answer_relevancy.mean():.4f}", f"- Average Faithfulness: {eval_df.faithfulness.mean():.4f}", f"- Average Context Precision: {eval_df.context_precision.mean():.4f}", f"- Evaluation Pass Rate: {(eval_df.status=='PASS').mean()*100:.2f}%"]
        if safety_df is not None and len(safety_df):
            lines += ['', '## Safety Assessment', '', f"- Safety tests: {len(safety_df)}", f"- Safety pass rate: {(safety_df.status=='PASS').mean()*100:.2f}%", f"- Failed safety cases: {(safety_df.status=='FAIL').sum()}"]
        if regression_df is not None and len(regression_df):
            lines += ['', '## Regression Results', '', f"- Regression cases: {len(regression_df)}", f"- Regression pass rate: {(regression_df.status=='PASS').mean()*100:.2f}%"]
        final='Conditionally Ready'
        if eval_df is not None and len(eval_df) and (eval_df.status=='PASS').all() and (safety_df is None or (safety_df.status=='PASS').all()): final='Production Ready'
        if safety_df is not None and (safety_df.status=='FAIL').any(): final='Not Ready'
        lines += ['', '## Acceptance Thresholds', '', f"- Relevance >= {THRESHOLDS['relevance']}", f"- Faithfulness >= {THRESHOLDS['faithfulness']}", f"- Context Precision >= {THRESHOLDS['precision']}", '', '## Final Recommendation', '', f'**{final}**', '', 'This recommendation is based on the measured framework outputs. It should not be interpreted as clinical validation or a substitute for clinical governance.']
        (OUTPUT_DIR/'final_benchmark_report.md').write_text('\n'.join(lines), encoding='utf-8')

    def export_excel(self):
        xlsx=OUTPUT_DIR/'clinical_llm_evaluation_report.xlsx'
        with pd.ExcelWriter(xlsx, engine='openpyxl') as writer:
            for name in ['evaluation_results','safety_results','regression_results']:
                p=OUTPUT_DIR/f'{name}.csv'
                if p.exists(): pd.read_csv(p).to_excel(writer, sheet_name=name[:31], index=False)
        return xlsx
