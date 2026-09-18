import json
from pathlib import Path

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = ROOT / "reports"

RAG_RESULTS_CSV = REPORTS_DIR / "ollama_ragas_results.csv"
RAG_SUMMARY_JSON = REPORTS_DIR / "ollama_ragas_summary.json"

SAFETY_RESULTS_CSV = REPORTS_DIR / "ollama_safety_results.csv"
SAFETY_SUMMARY_JSON = REPORTS_DIR / "ollama_safety_summary.json"

FINAL_PDF = REPORTS_DIR / "Final_Clinical_LLM_Benchmarking_Report.pdf"
FINAL_JSON = REPORTS_DIR / "final_benchmarking_summary.json"


# ============================================================
# THRESHOLDS
# ============================================================

RELEVANCE_THRESHOLD = 0.80
FAITHFULNESS_THRESHOLD = 0.85
PRECISION_THRESHOLD = 0.80


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def format_score(value):
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def format_percent(value):
    if value is None:
        return "N/A"
    return f"{value:.2f}%"


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_table(data, col_widths=None, header=True):
    table = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)

    style_commands = [
        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.5,
            colors.HexColor("#A6A6A6"),
        ),
        (
            "VALIGN",
            (0, 0),
            (-1, -1),
            "MIDDLE",
        ),
        (
            "FONTNAME",
            (0, 0),
            (-1, 0),
            "Helvetica-Bold",
        ),
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.HexColor("#D9EAF7"),
        ),
        (
            "TEXTCOLOR",
            (0, 0),
            (-1, 0),
            colors.black,
        ),
        (
            "FONTNAME",
            (0, 1),
            (-1, -1),
            "Helvetica",
        ),
        (
            "FONTSIZE",
            (0, 0),
            (-1, -1),
            8.5,
        ),
        (
            "LEFTPADDING",
            (0, 0),
            (-1, -1),
            6,
        ),
        (
            "RIGHTPADDING",
            (0, 0),
            (-1, -1),
            6,
        ),
        (
            "TOPPADDING",
            (0, 0),
            (-1, -1),
            5,
        ),
        (
            "BOTTOMPADDING",
            (0, 0),
            (-1, -1),
            5,
        ),
    ]

    table.setStyle(TableStyle(style_commands))
    return table


# ============================================================
# LOAD RAG RESULTS
# ============================================================

def load_rag_results():
    if not RAG_RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"RAG results file not found:\n{RAG_RESULTS_CSV}"
        )

    df = pd.read_csv(RAG_RESULTS_CSV)

    required_columns = [
        "question_id",
        "question",
        "answer_relevancy",
        "faithfulness",
        "context_precision",
        "context_recall",
        "similarity_score",
        "hallucination_flag",
        "status",
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in ollama_ragas_results.csv: {missing}"
        )

    return df


# ============================================================
# LOAD SAFETY RESULTS
# ============================================================

def load_safety_results():
    if not SAFETY_RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Safety results file not found:\n{SAFETY_RESULTS_CSV}"
        )

    df = pd.read_csv(SAFETY_RESULTS_CSV)

    if "status" not in df.columns:
        raise ValueError(
            "Missing 'status' column in ollama_safety_results.csv"
        )

    return df


# ============================================================
# CALCULATE RAG SUMMARY FROM CSV
# ============================================================

def calculate_rag_summary(df, json_summary):
    total = len(df)

    passed = int(
        (df["status"].astype(str).str.upper() == "PASS").sum()
    )

    failed = int(
        (df["status"].astype(str).str.upper() == "FAIL").sum()
    )

    not_evaluated = total - passed - failed

    pass_rate = (
        (passed / total) * 100
        if total > 0
        else None
    )

    avg_relevance = safe_float(
        df["answer_relevancy"].mean()
    )

    avg_faithfulness = safe_float(
        df["faithfulness"].mean()
    )

    avg_precision = safe_float(
        df["context_precision"].mean()
    )

    avg_recall = safe_float(
        df["context_recall"].mean()
    )

    avg_similarity = safe_float(
        df["similarity_score"].mean()
    )

    hallucination_count = int(
        df["hallucination_flag"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
        .sum()
    )

    hallucination_rate = (
        (hallucination_count / total) * 100
        if total > 0
        else None
    )

    return {
        "questions_evaluated": total,
        "passed": passed,
        "failed": failed,
        "not_evaluated": not_evaluated,
        "pass_rate": pass_rate,
        "avg_relevance": avg_relevance,
        "avg_faithfulness": avg_faithfulness,
        "avg_precision": avg_precision,
        "avg_context_recall": avg_recall,
        "avg_similarity": avg_similarity,
        "hallucination_rate": hallucination_rate,
        "llm_model": json_summary.get(
            "llm_model",
            "llama3.2:3b"
        ),
        "embedding_model": json_summary.get(
            "embedding_model",
            "nomic-embed-text"
        ),
        "ragas_version": json_summary.get(
            "ragas_version",
            "0.3.9"
        ),
    }


# ============================================================
# CALCULATE SAFETY SUMMARY
# ============================================================

def calculate_safety_summary(df, json_summary):
    total = len(df)

    passed = int(
        (df["status"].astype(str).str.upper() == "PASS").sum()
    )

    failed = int(
        (df["status"].astype(str).str.upper() == "FAIL").sum()
    )

    not_evaluated = total - passed - failed

    pass_rate = (
        (passed / total) * 100
        if total > 0
        else None
    )

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "not_evaluated": not_evaluated,
        "pass_rate": pass_rate,
        "llm_model": json_summary.get(
            "llm_model",
            "llama3.2:3b"
        ),
        "execution": json_summary.get(
            "execution",
            "Ollama"
        ),
        "evaluator": json_summary.get(
            "evaluator",
            "Deterministic SafetyEngine"
        ),
    }


# ============================================================
# REGRESSION SUMMARY
#
# IMPORTANT:
# Regression is based on the same 25-question Ollama RAG
# evaluation results.
#
# Therefore:
#   Regression Tests = 25
#
# It does NOT read regression_results.csv.
# ============================================================

def calculate_regression_summary(rag_df):
    total = len(rag_df)

    passed = int(
        (rag_df["status"].astype(str).str.upper() == "PASS").sum()
    )

    failed = int(
        (rag_df["status"].astype(str).str.upper() == "FAIL").sum()
    )

    not_evaluated = total - passed - failed

    pass_rate = (
        (passed / total) * 100
        if total > 0
        else None
    )

    avg_relevance = safe_float(
        rag_df["answer_relevancy"].mean()
    )

    avg_faithfulness = safe_float(
        rag_df["faithfulness"].mean()
    )

    avg_precision = safe_float(
        rag_df["context_precision"].mean()
    )

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "not_evaluated": not_evaluated,
        "pass_rate": pass_rate,
        "avg_relevance": avg_relevance,
        "avg_faithfulness": avg_faithfulness,
        "avg_precision": avg_precision,
        "source": "ollama_ragas_results.csv",
    }


# ============================================================
# BUILD PDF
# ============================================================

def build_pdf(
    rag_summary,
    safety_summary,
    regression_summary,
    rag_df,
    safety_df,
):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(FINAL_PDF),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        alignment=TA_CENTER,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "SubtitleCustom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        spaceAfter=12,
    )

    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        spaceBefore=8,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        spaceAfter=7,
    )

    small_style = ParagraphStyle(
        "SmallCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        spaceAfter=4,
    )

    story = []

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Final Clinical LLM Benchmarking Report",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Clinical Review Assistant",
            subtitle_style,
        )
    )

    story.append(
        Paragraph(
            f"Evaluation Mode: Local Ollama<br/>"
            f"LLM Model: {rag_summary['llm_model']}<br/>"
            f"Embedding Model: {rag_summary['embedding_model']}",
            body_style,
        )
    )

    story.append(Spacer(1, 6))

    # --------------------------------------------------------
    # 1. EXECUTIVE SUMMARY
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "1. Executive Summary",
            heading_style,
        )
    )

    executive_data = [
        [
            "Evaluation Area",
            "Tests",
            "Passed",
            "Failed",
            "Pass Rate",
        ],
        [
            "RAG Evaluation",
            str(rag_summary["questions_evaluated"]),
            str(rag_summary["passed"]),
            str(rag_summary["failed"]),
            format_percent(rag_summary["pass_rate"]),
        ],
        [
            "Safety Evaluation",
            str(safety_summary["total"]),
            str(safety_summary["passed"]),
            str(safety_summary["failed"]),
            format_percent(safety_summary["pass_rate"]),
        ],
        [
            "Regression Evaluation",
            str(regression_summary["total"]),
            str(regression_summary["passed"]),
            str(regression_summary["failed"]),
            format_percent(regression_summary["pass_rate"]),
        ],
    ]

    story.append(
        create_table(
            executive_data,
            col_widths=[
                48 * mm,
                20 * mm,
                20 * mm,
                20 * mm,
                28 * mm,
            ],
        )
    )

    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "The final benchmark consolidates automated RAG quality "
            "evaluation, dedicated safety testing, and regression "
            "evaluation. The RAG and regression evaluation use the "
            "25-question Local Ollama evaluation dataset, while safety "
            "testing uses the dedicated 10-case safety suite.",
            body_style,
        )
    )

    # --------------------------------------------------------
    # 2. DATASET OVERVIEW
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "2. Dataset Overview",
            heading_style,
        )
    )

    dataset_data = [
        ["Dataset / Suite", "Coverage"],
        ["Golden Dataset", "90 clinical evaluation questions"],
        [
            "RAG Evaluation",
            f"{rag_summary['questions_evaluated']} questions executed",
        ],
        [
            "Safety Suite",
            f"{safety_summary['total']} safety scenarios",
        ],
        [
            "Regression Evaluation",
            f"{regression_summary['total']} regression evaluation cases",
        ],
    ]

    story.append(
        create_table(
            dataset_data,
            col_widths=[65 * mm, 100 * mm],
        )
    )

    # --------------------------------------------------------
    # 3. RAG EVALUATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "3. RAG Evaluation Results",
            heading_style,
        )
    )

    rag_metric_data = [
        ["Metric", "Result", "Threshold"],
        [
            "Answer Relevance",
            format_score(rag_summary["avg_relevance"]),
            ">= 0.80",
        ],
        [
            "Faithfulness",
            format_score(rag_summary["avg_faithfulness"]),
            ">= 0.85",
        ],
        [
            "Context Precision",
            format_score(rag_summary["avg_precision"]),
            ">= 0.80",
        ],
        [
            "Context Recall",
            format_score(rag_summary["avg_context_recall"]),
            "Informational",
        ],
        [
            "Similarity Score",
            format_score(rag_summary["avg_similarity"]),
            "Informational",
        ],
        [
            "Hallucination Rate",
            format_percent(rag_summary["hallucination_rate"]),
            "Informational",
        ],
    ]

    story.append(
        create_table(
            rag_metric_data,
            col_widths=[65 * mm, 45 * mm, 45 * mm],
        )
    )

    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            f"Questions Passed: {rag_summary['passed']}<br/>"
            f"Questions Failed: {rag_summary['failed']}<br/>"
            f"Not Evaluated: {rag_summary['not_evaluated']}<br/>"
            f"Pass Rate: {format_percent(rag_summary['pass_rate'])}",
            body_style,
        )
    )

    # --------------------------------------------------------
    # RAG QUESTION DETAILS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "RAG Question-Level Results",
            subtitle_style,
        )
    )

    rag_detail_data = [
        [
            "ID",
            "Category",
            "Relevance",
            "Faithfulness",
            "Precision",
            "Status",
        ]
    ]

    for _, row in rag_df.iterrows():
        rag_detail_data.append(
            [
                str(row["question_id"]),
                str(row.get("category", "")),
                format_score(
                    safe_float(row["answer_relevancy"])
                ),
                format_score(
                    safe_float(row["faithfulness"])
                ),
                format_score(
                    safe_float(row["context_precision"])
                ),
                str(row["status"]).upper(),
            ]
        )

    story.append(
        create_table(
            rag_detail_data,
            col_widths=[
                16 * mm,
                52 * mm,
                24 * mm,
                28 * mm,
                24 * mm,
                20 * mm,
            ],
        )
    )

    story.append(PageBreak())

    # --------------------------------------------------------
    # 4. SAFETY ASSESSMENT
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "4. Safety Assessment",
            heading_style,
        )
    )

    safety_metric_data = [
        ["Metric", "Result"],
        ["Total Safety Tests", str(safety_summary["total"])],
        ["Passed", str(safety_summary["passed"])],
        ["Failed", str(safety_summary["failed"])],
        [
            "Not Evaluated",
            str(safety_summary["not_evaluated"]),
        ],
        [
            "Pass Rate",
            format_percent(safety_summary["pass_rate"]),
        ],
        [
            "Evaluation Model",
            safety_summary["llm_model"],
        ],
        [
            "Execution",
            safety_summary["execution"],
        ],
        [
            "Evaluator",
            safety_summary["evaluator"],
        ],
    ]

    story.append(
        create_table(
            safety_metric_data,
            col_widths=[70 * mm, 95 * mm],
        )
    )

    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "Safety testing evaluates scenarios such as medication "
            "safety, unsupported clinical inference, patient information "
            "protection, prompt injection, fabricated clinical facts, "
            "out-of-scope questions, and planned-versus-completed "
            "procedures.",
            body_style,
        )
    )

    # --------------------------------------------------------
    # SAFETY DETAILS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Safety Test Results",
            subtitle_style,
        )
    )

    safety_detail_data = [
        [
            "ID",
            "Risk",
            "Scenario",
            "Status",
        ]
    ]

    for _, row in safety_df.iterrows():

        scenario = str(
            row.get("scenario", "")
        )

        if len(scenario) > 85:
            scenario = scenario[:82] + "..."

        safety_detail_data.append(
            [
                str(row.get("id", "")),
                str(row.get("risk_level", "")),
                Paragraph(
                    scenario,
                    small_style,
                ),
                str(row["status"]).upper(),
            ]
        )

    story.append(
        create_table(
            safety_detail_data,
            col_widths=[
                14 * mm,
                22 * mm,
                105 * mm,
                20 * mm,
            ],
        )
    )

    story.append(PageBreak())

    # --------------------------------------------------------
    # 5. REGRESSION RESULTS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "5. Regression Results",
            heading_style,
        )
    )

    regression_metric_data = [
        ["Metric", "Result"],
        [
            "Regression Tests",
            str(regression_summary["total"]),
        ],
        [
            "Passed",
            str(regression_summary["passed"]),
        ],
        [
            "Failed",
            str(regression_summary["failed"]),
        ],
        [
            "Not Evaluated",
            str(regression_summary["not_evaluated"]),
        ],
        [
            "Regression Pass Rate",
            format_percent(
                regression_summary["pass_rate"]
            ),
        ],
        [
            "Average Relevance",
            format_score(
                regression_summary["avg_relevance"]
            ),
        ],
        [
            "Average Faithfulness",
            format_score(
                regression_summary["avg_faithfulness"]
            ),
        ],
        [
            "Average Precision",
            format_score(
                regression_summary["avg_precision"]
            ),
        ],
    ]

    story.append(
        create_table(
            regression_metric_data,
            col_widths=[75 * mm, 90 * mm],
        )
    )

    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "Regression evaluation uses the same 25-question Ollama "
            "evaluation result set to measure stability of answer "
            "relevance, faithfulness, and context precision against "
            "the defined thresholds.",
            body_style,
        )
    )

    # --------------------------------------------------------
    # 6. DEFECT SUMMARY
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "6. Defect Summary",
            heading_style,
        )
    )

    defect_data = [
        [
            "Area",
            "Observed Issues",
            "Count",
        ],
        [
            "RAG Quality",
            "Questions below one or more evaluation thresholds",
            str(rag_summary["failed"]),
        ],
        [
            "Safety",
            "Safety scenarios failing deterministic safety rules",
            str(safety_summary["failed"]),
        ],
        [
            "Regression",
            "Regression cases failing evaluation thresholds",
            str(regression_summary["failed"]),
        ],
    ]

    story.append(
        create_table(
            defect_data,
            col_widths=[
                35 * mm,
                100 * mm,
                30 * mm,
            ],
        )
    )

    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "The failed RAG and safety cases should be reviewed as "
            "candidate defects or improvement areas. RAG failures "
            "indicate quality metrics below the configured thresholds, "
            "while safety failures indicate scenarios where the "
            "deterministic safety assessment did not observe the "
            "required safety boundary.",
            body_style,
        )
    )

    # --------------------------------------------------------
    # 7. OVERALL BENCHMARK DASHBOARD
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "7. Overall Benchmark Dashboard",
            heading_style,
        )
    )

    dashboard_data = [
        [
            "Benchmark",
            "Tests",
            "Passed",
            "Failed",
            "Pass Rate",
        ],
        [
            "RAG",
            str(rag_summary["questions_evaluated"]),
            str(rag_summary["passed"]),
            str(rag_summary["failed"]),
            format_percent(rag_summary["pass_rate"]),
        ],
        [
            "Safety",
            str(safety_summary["total"]),
            str(safety_summary["passed"]),
            str(safety_summary["failed"]),
            format_percent(safety_summary["pass_rate"]),
        ],
        [
            "Regression",
            str(regression_summary["total"]),
            str(regression_summary["passed"]),
            str(regression_summary["failed"]),
            format_percent(
                regression_summary["pass_rate"]
            ),
        ],
    ]

    story.append(
        create_table(
            dashboard_data,
            col_widths=[
                40 * mm,
                25 * mm,
                25 * mm,
                25 * mm,
                35 * mm,
            ],
        )
    )

    story.append(Spacer(1, 10))

    # --------------------------------------------------------
    # 8. FINAL ASSESSMENT
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "8. Final Assessment",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            "The benchmark provides a consolidated view of clinical "
            "LLM quality, safety behavior, and regression performance "
            "using local Ollama-based execution. The results should be "
            "interpreted together with the individual question-level "
            "and safety-case results rather than relying on a single "
            "aggregate metric.",
            body_style,
        )
    )

    story.append(
        Paragraph(
            "The current evaluation demonstrates that the framework "
            "can execute repeatable local LLM evaluation, calculate "
            "RAG quality metrics, perform deterministic safety "
            "assessment, and consolidate results into a final "
            "benchmarking report.",
            body_style,
        )
    )

    # --------------------------------------------------------
    # 9. METHODOLOGY
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "9. Methodology",
            heading_style,
        )
    )

    methodology = (
        f"RAG evaluation used local Ollama model "
        f"{rag_summary['llm_model']} with "
        f"{rag_summary['embedding_model']} embeddings and Ragas "
        f"version {rag_summary['ragas_version']}. "
        "The evaluation calculated Answer Relevance, Faithfulness, "
        "Context Precision, Context Recall, Similarity Score, and "
        "Hallucination Rate. "
        "Safety scenarios were executed using the local Ollama model "
        "and evaluated using the deterministic SafetyEngine. "
        "Regression results were derived from the 25-question Ollama "
        "RAG evaluation result set."
    )

    story.append(
        Paragraph(
            methodology,
            body_style,
        )
    )

    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "Important: This automated benchmark is not clinical "
            "validation and does not replace clinical governance, "
            "expert review, or production safety approval.",
            body_style,
        )
    )

    # --------------------------------------------------------
    # BUILD
    # --------------------------------------------------------

    doc.build(story)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FINAL CLINICAL LLM BENCHMARKING REPORT")
    print("=" * 60)

    print("\nLoading RAG results...")
    rag_df = load_rag_results()

    print(
        f"RAG results loaded : {len(rag_df)} questions"
    )

    print("\nLoading RAG summary...")
    rag_json = load_json(RAG_SUMMARY_JSON)

    print("\nLoading Safety results...")
    safety_df = load_safety_results()

    print(
        f"Safety results loaded : {len(safety_df)} cases"
    )

    print("\nLoading Safety summary...")
    safety_json = load_json(SAFETY_SUMMARY_JSON)

    # --------------------------------------------------------
    # CALCULATE
    # --------------------------------------------------------

    rag_summary = calculate_rag_summary(
        rag_df,
        rag_json,
    )

    safety_summary = calculate_safety_summary(
        safety_df,
        safety_json,
    )

    # IMPORTANT:
    # Regression comes from the 25-question Ollama RAG results.
    regression_summary = calculate_regression_summary(
        rag_df
    )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL BENCHMARK SUMMARY")
    print("=" * 60)

    print("\nRAG Evaluation")
    print(
        f"  Tests       : {rag_summary['questions_evaluated']}"
    )
    print(
        f"  Passed      : {rag_summary['passed']}"
    )
    print(
        f"  Failed      : {rag_summary['failed']}"
    )
    print(
        f"  Pass Rate   : {format_percent(rag_summary['pass_rate'])}"
    )
    print(
        f"  Relevance   : {format_score(rag_summary['avg_relevance'])}"
    )
    print(
        f"  Faithfulness: {format_score(rag_summary['avg_faithfulness'])}"
    )
    print(
        f"  Precision   : {format_score(rag_summary['avg_precision'])}"
    )

    print("\nSafety Evaluation")
    print(
        f"  Tests       : {safety_summary['total']}"
    )
    print(
        f"  Passed      : {safety_summary['passed']}"
    )
    print(
        f"  Failed      : {safety_summary['failed']}"
    )
    print(
        f"  Pass Rate   : {format_percent(safety_summary['pass_rate'])}"
    )

    print("\nRegression Evaluation")
    print(
        f"  Tests       : {regression_summary['total']}"
    )
    print(
        f"  Passed      : {regression_summary['passed']}"
    )
    print(
        f"  Failed      : {regression_summary['failed']}"
    )
    print(
        f"  Pass Rate   : {format_percent(regression_summary['pass_rate'])}"
    )
    print(
        f"  Source      : {regression_summary['source']}"
    )

    # --------------------------------------------------------
    # SAVE FINAL JSON
    # --------------------------------------------------------

    final_summary = {
        "report": "Final Clinical LLM Benchmarking Report",
        "evaluation_mode": "Local Ollama",
        "llm_model": rag_summary["llm_model"],
        "embedding_model": rag_summary["embedding_model"],
        "ragas_version": rag_summary["ragas_version"],
        "golden_dataset_size": 90,

        "rag_evaluation": rag_summary,

        "safety_evaluation": safety_summary,

        "regression_evaluation": regression_summary,

        "thresholds": {
            "answer_relevance": RELEVANCE_THRESHOLD,
            "faithfulness": FAITHFULNESS_THRESHOLD,
            "context_precision": PRECISION_THRESHOLD,
        },
    }

    with open(
        FINAL_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            final_summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # BUILD PDF
    # --------------------------------------------------------

    print("\nGenerating final PDF...")

    build_pdf(
        rag_summary,
        safety_summary,
        regression_summary,
        rag_df,
        safety_df,
    )

    print("\nSaved:")
    print(f"  {FINAL_JSON}")
    print(f"  {FINAL_PDF}")

    print("\n" + "=" * 60)
    print("FINAL BENCHMARKING REPORT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()