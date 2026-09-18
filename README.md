# Clinical Review Assistant - LLM Evaluation Framework

A clean, standalone AI Quality Engineering framework for evaluating the Clinical Review Assistant.

## Architecture

Test Data -> Chatbot Executor -> Evaluation Engine -> Safety Engine -> Results -> Reports/Dashboard -> Regression Suite

## What is included

- 90-case Golden Dataset
- Safety/adversarial test suite
- Playwright executor for the public Streamlit chatbot
- Optional HTTP/JSON executor for an API-backed version
- Local knowledge-base retriever used when the chatbot does not expose retrieved contexts
- Ragas evaluation for Faithfulness, Answer Relevancy and Context Precision
- Optional Context Recall, answer similarity and custom hallucination indicators
- Safety rule checks plus LLM-based safety judging
- Defect generation and severity/priority classification
- Regression runner
- CSV/Excel/JSON/Markdown reports
- Streamlit dashboard
- Pytest smoke tests

## Important note about the demo application

The public Streamlit application may not expose its internal retrieved context through the browser. The framework therefore supports two context modes:

1. `local_kb`: retrieves evidence from `data/knowledge_base.md`.
2. `api`: uses contexts returned by a JSON API if the application provides them.

For a browser-only Streamlit deployment, `local_kb` is the reproducible fallback. It is explicitly reported in the run metadata so the limitation is visible in a demo.

## Setup

1. Create a fresh Python virtual environment.
2. Install `requirements.txt`.
3. Run `playwright install chromium`.
4. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
5. Run the full evaluation:

```bash
python -m src.run_all
```

6. Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

## Useful commands

```bash
python -m src.run_evaluation
python -m src.run_safety
python -m src.run_regression
python -m src.generate_reports
pytest -q
```

## Acceptance thresholds

- Relevance >= 0.80
- Faithfulness >= 0.85
- Context Precision >= 0.80

These thresholds are configurable in `config/settings.yaml`.
