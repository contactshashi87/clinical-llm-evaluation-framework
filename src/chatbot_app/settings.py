import os
import yaml
from dotenv import load_dotenv

from ..shared.paths import ROOT

load_dotenv(ROOT / ".env")


with open(
    ROOT / "config" / "chatbot_app.yaml",
    encoding="utf-8"
) as f:
    CFG = yaml.safe_load(f)


# ============================================================
# Application
# ============================================================

APP_URL = os.getenv(
    "CLINICAL_APP_URL",
    CFG["application"]["url"]
)

CHATBOT_MODE = os.getenv(
    "CHATBOT_MODE",
    CFG["application"]["mode"]
)

CONTEXT_MODE = os.getenv(
    "CONTEXT_MODE",
    CFG["application"]["context_mode"]
)

HEADLESS = (
    os.getenv(
        "BROWSER_HEADLESS",
        str(CFG["application"]["headless"])
    ).lower()
    == "true"
)

TIMEOUT = int(
    os.getenv(
        "REQUEST_TIMEOUT_SECONDS",
        CFG["application"]["timeout_seconds"]
    )
)


# ============================================================
# Evaluation
# ============================================================

EVALUATION = CFG["evaluation"]

EVAL_MODEL = os.getenv(
    "EVALUATOR_MODEL",
    CFG["evaluation"]["model"]
)

EMBED_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    CFG["evaluation"]["embedding_model"]
)

THRESHOLDS = CFG["evaluation"]["thresholds"]


# ============================================================
# Execution
# ============================================================

RETRY_COUNT = int(
    CFG["execution"]["retry_count"]
)

PAUSE = float(
    CFG["execution"]["pause_between_questions_seconds"]
)


# ============================================================
# Reporting
# ============================================================

OUTPUT_DIR = ROOT / CFG["reporting"]["output_dir"]

OUTPUT_DIR.mkdir(
    exist_ok=True
)