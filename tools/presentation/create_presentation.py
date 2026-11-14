from pptx import Presentation
from pptx.util import Inches, Pt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
prs=Presentation(); prs.slide_width=Inches(13.333); prs.slide_height=Inches(7.5)
slides=[
('Clinical Review Assistant','LLM Evaluation Framework\n10-minute AI Quality Engineering demo'),
('Objective','Evaluate answer quality, faithfulness, precision, safety, reliability and regression stability.'),
('Framework Architecture','Test Data → Chatbot Executor → Evaluation Engine → Safety Engine → Results → Reports/Dashboard → Regression Suite'),
('Golden Dataset','90 scenarios\n20 Factual Clinical\n15 Medication\n15 Procedure\n15 Policy/Guideline\n10 Edge Cases\n10 Out-of-Scope\n5 Adversarial'),
('Automated Evaluation','Ragas metrics\n• Answer Relevance ≥ 0.80\n• Faithfulness ≥ 0.85\n• Context Precision ≥ 0.80\nOptional similarity and hallucination indicators'),
('Safety Testing','Clinical safety\nHallucination / out-of-KB\nPrompt injection\nSensitive patient-data exposure\nUnsafe medication recommendations'),
('Defect Reporting','Functional defects\nAI defects\nSafety defects\nEvidence + severity + priority + expected vs actual'),
('Regression Testing','Critical clinical questions\nHigh-risk scenarios\nSafety tests\nPrompt injection\nMulti-turn conversations'),
('Dashboard & Reports','Per-question scores\nPass/fail\nSafety findings\nDefect log\nRegression pass rate\nExecutive benchmark report'),
('Final Recommendation','Production Ready / Conditionally Ready / Not Ready\nBased on measured evidence and documented risks.')]
for title,body in slides:
    slide=prs.slides.add_slide(prs.slide_layouts[5])
    tb=slide.shapes.add_textbox(Inches(0.8),Inches(0.7),Inches(11.8),Inches(1.0)); p=tb.text_frame.paragraphs[0]; p.text=title; p.font.size=Pt(30); p.font.bold=True
    bb=slide.shapes.add_textbox(Inches(1.0),Inches(2.0),Inches(11.2),Inches(4.5)); tf=bb.text_frame; tf.word_wrap=True; p=tf.paragraphs[0]; p.text=body; p.font.size=Pt(23)
OUTPUT = ROOT / 'docs' / 'presentation' / 'Clinical_LLM_Evaluation_Demo_Template.pptx'
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUTPUT)
