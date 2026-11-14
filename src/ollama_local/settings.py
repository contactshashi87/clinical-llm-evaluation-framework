import yaml

from ..shared.paths import ROOT


with open(ROOT / "config" / "ollama_local.yaml", encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)

OLLAMA_HOST = CONFIG["host"]
LLM_MODEL = CONFIG["model"]
EMBEDDING_MODEL = CONFIG["embedding_model"]
QUESTIONS_TO_EVALUATE = CONFIG["questions_to_evaluate"]