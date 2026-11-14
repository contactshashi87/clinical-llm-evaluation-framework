from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def test_golden_dataset_has_90_cases_and_required_categories():
    df=pd.read_csv(ROOT/'data/shared/golden_dataset.csv')
    assert len(df)==90
    expected={'Factual Clinical Question':20,'Medication Question':15,'Procedure Question':15,'Policy/Guideline Question':15,'Edge Case':10,'Out-of-Scope Question':10,'Adversarial Prompt':5}
    assert df.category.value_counts().to_dict()==expected

def test_thresholds_are_present():
    text=(ROOT/'config/chatbot_app.yaml').read_text()
    assert 'relevance: 0.80' in text
    assert 'faithfulness: 0.85' in text
    assert 'precision: 0.80' in text
