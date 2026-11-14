# Clinical Review Assistant - LLM Evaluation Framework

A clean, standalone AI Quality Engineering framework for evaluating the Clinical Review Assistant.

## Architecture

The framework has two independent evaluation targets:

- `src/chatbot_app/` tests the deployed Clinical Review Assistant through Playwright, then evaluates its responses.
- `src/ollama_local/` tests responses generated directly by the local Ollama model and its local RAG pipeline.
- Shared datasets and `knowledge_base.md` are under `data/shared/`; chatbot-only multi-turn cases are under `data/chatbot_app/`.
- Deterministic safety rules and retrieval code are shared under `src/`.
- Results are written separately to `reports/chatbot_app/` and `reports/ollama_local/`.

## What is included

- 90-case Golden Dataset
- Safety/adversarial test suite
- Playwright executor for the public Streamlit chatbot
- Local knowledge-base retriever used when the chatbot does not expose retrieved contexts
- Local Ollama RAG and safety evaluation as a separate test target
- Ragas evaluation for Faithfulness, Answer Relevancy and Context Precision
- Optional Context Recall, answer similarity and custom hallucination indicators
- Safety rule checks plus LLM-based safety judging
- Defect generation and severity/priority classification
- Regression runner
- CSV/Excel/JSON/Markdown reports
- Streamlit dashboard
- Pytest smoke tests

## Important note about the demo application

The public Streamlit application may not expose its internal retrieved context through the browser. When a chatbot regression result has no contexts, the framework uses `data/shared/knowledge_base.md` as a local fallback and records that context source.

## Setup

1. Create a fresh Python virtual environment.
2. Install `requirements.txt`.
3. Run `playwright install chromium`.
4. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
5. Run either evaluation target:

```bash
python -m src.chatbot_app.run_all
python -m src.ollama_local.run_all
```

6. Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

## Useful commands

```bash
python -m src.chatbot_app.run_evaluation
python -m src.chatbot_app.run_safety
python -m src.chatbot_app.run_regression
python -m src.chatbot_app.generate_reports
python -m src.ollama_local.ollama_rag_evaluation
python -m src.ollama_local.ollama_safety_evaluation
pytest -q
```

## Acceptance thresholds

- Relevance >= 0.80
- Faithfulness >= 0.85
- Context Precision >= 0.80

These thresholds are configurable in `config/chatbot_app.yaml`.
