import streamlit as st
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports'/'chatbot_app'
st.set_page_config(page_title='Clinical LLM Evaluation Dashboard', layout='wide')
st.title('Clinical Review Assistant — LLM Evaluation Dashboard')
st.caption('AI Quality Engineering evaluation framework')

def load(name):
    p=REPORTS/name
    return pd.read_csv(p) if p.exists() else None

e=load('evaluation_results.csv'); s=load('safety_results.csv'); r=load('regression_results.csv'); d=load('defect_log.csv')

if e is None:
    st.warning('No chatbot-app results found. Run `python -m src.chatbot_app.run_all` first.')
    st.stop()

c1,c2,c3,c4=st.columns(4)
c1.metric('Evaluation pass rate', f"{(e.status=='PASS').mean()*100:.1f}%")
c2.metric('Avg relevance', f"{e.answer_relevancy.mean():.3f}")
c3.metric('Avg faithfulness', f"{e.faithfulness.mean():.3f}")
c4.metric('Avg precision', f"{e.context_precision.mean():.3f}")

st.subheader('Evaluation Results')
st.dataframe(e[['id','category','question','answer_relevancy','faithfulness','context_precision','status']], use_container_width=True)

if s is not None:
    st.subheader('Safety Results')
    sc1,sc2=st.columns(2)
    sc1.metric('Safety pass rate', f"{(s.status=='PASS').mean()*100:.1f}%")
    sc2.metric('Safety failures', int((s.status=='FAIL').sum()))
    st.dataframe(s[['id','category','risk_level','scenario','status','actual_result']], use_container_width=True)

if d is not None:
    st.subheader('Defect Log')
    st.dataframe(d, use_container_width=True)

if r is not None:
    st.subheader('Regression Suite')
    st.metric('Regression pass rate', f"{(r.status=='PASS').mean()*100:.1f}%")
    st.dataframe(r[['id','category','question','answer_relevancy','faithfulness','context_precision','status']], use_container_width=True)

report=REPORTS/'final_benchmark_report.md'
if report.exists():
    st.subheader('Final Benchmarking Report')
    st.markdown(report.read_text(encoding='utf-8'))
