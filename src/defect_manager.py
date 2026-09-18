import pandas as pd
from .settings import OUTPUT_DIR

def generate_defects(eval_df=None, safety_df=None):
    defects=[]; n=1
    if eval_df is not None:
        for _,r in eval_df[eval_df.status=='FAIL'].iterrows():
            failed=[]
            if r.answer_relevancy < .80: failed.append('Answer Relevance')
            if r.faithfulness < .85: failed.append('Faithfulness')
            if r.context_precision < .80: failed.append('Context Precision')
            sev='High' if r.risk_level in {'high','critical'} else 'Medium'
            defects.append({'bug_id':f'LLM-{n:03d}','title':f'LLM response failed: {", ".join(failed)}','severity':sev,'priority':'P1' if sev=='High' else 'P2','category':'AI Defect','steps':f'Ask: {r.question}','question_asked':r.question,'actual_response':r.answer,'expected_response':r.expected_answer,'evidence':f"Relevance={r.answer_relevancy:.3f}; Faithfulness={r.faithfulness:.3f}; Precision={r.context_precision:.3f}"}); n+=1
    if safety_df is not None:
        for _,r in safety_df[safety_df.status=='FAIL'].iterrows():
            sev='Critical' if r.risk_level=='critical' else 'High'
            defects.append({'bug_id':f'SAF-{n:03d}','title':'Safety control failure','severity':sev,'priority':'P0' if sev=='Critical' else 'P1','category':'Safety Defect','steps':f"Send scenario: {r.scenario}",'question_asked':r.scenario,'actual_response':r.actual_result,'expected_response':r.expected_result,'evidence':r.judge_reason}); n+=1
    df=pd.DataFrame(defects)
    df.to_csv(OUTPUT_DIR/'defect_log.csv',index=False)
    return df
