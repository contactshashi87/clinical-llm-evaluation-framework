import asyncio
from pathlib import Path

import pandas as pd

from .settings import ROOT, OUTPUT_DIR
from .chatbot_executor import ChatbotExecutor
from ..shared.safety_engine import SafetyEngine
from .reporting import ReportManager


# ============================================================
# SAFETY PDF REPORT
# ============================================================

def generate_safety_pdf(result: pd.DataFrame):
    """
    Generate a plain Safety Test PDF from the current execution
    results.

    The PDF is generated automatically every time the safety suite
    is executed.
    """

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    reports_dir = OUTPUT_DIR
    reports_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    report_pdf = (
        reports_dir /
        "Safety_Test_Report.pdf"
    )

    # --------------------------------------------------------
    # Document
    # --------------------------------------------------------

    document = SimpleDocTemplate(
        str(report_pdf),
        pagesize=landscape(A4),
        rightMargin=10 * mm,
        leftMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=12 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "SafetyTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "SafetySubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        spaceAfter=12,
    )

    heading_style = ParagraphStyle(
        "SafetyHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        spaceBefore=8,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "SafetyBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
    )

    small_style = ParagraphStyle(
        "SafetySmall",
        parent=body_style,
        fontSize=7,
        leading=8.5,
    )

    header_style = ParagraphStyle(
        "SafetyHeader",
        parent=body_style,
        fontName="Helvetica-Bold",
        fontSize=7,
        leading=8,
        alignment=TA_CENTER,
    )

    story = []

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Safety Test Report",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Clinical Review Assistant",
            subtitle_style
        )
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Safety Summary",
            heading_style
        )
    )

    total = len(result)

    passed = int(
        (
            result["status"]
            .astype(str)
            .str.upper()
            == "PASS"
        ).sum()
    )

    failed = int(
        (
            result["status"]
            .astype(str)
            .str.upper()
            == "FAIL"
        ).sum()
    )

    not_evaluated = int(
        (
            result["status"]
            .astype(str)
            .str.upper()
            == "NOT_EVALUATED"
        ).sum()
    )

    pass_rate = (
        (passed / total) * 100
        if total > 0
        else 0
    )

    summary_data = [
        [
            Paragraph(
                "<b>Total Scenarios</b>",
                body_style
            ),
            Paragraph(
                "<b>Passed</b>",
                body_style
            ),
            Paragraph(
                "<b>Failed</b>",
                body_style
            ),
            Paragraph(
                "<b>Not Evaluated</b>",
                body_style
            ),
            Paragraph(
                "<b>Pass Rate</b>",
                body_style
            ),
        ],
        [
            str(total),
            str(passed),
            str(failed),
            str(not_evaluated),
            f"{pass_rate:.2f}%",
        ],
    ]

    summary_table = Table(
        summary_data,
        colWidths=[
            35 * mm,
            30 * mm,
            30 * mm,
            35 * mm,
            35 * mm,
        ],
    )

    summary_table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8,
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
        )
    )

    story.append(summary_table)

    story.append(
        Spacer(1, 8 * mm)
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Safety Test Results",
            heading_style
        )
    )

    columns = [
        "id",
        "risk_level",
        "scenario",
        "expected_result",
        "actual_result",
        "executor_status",
        "rule_check",
        "judge_reason",
        "status",
    ]

    display_headers = [
        "ID",
        "Risk Level",
        "Scenario",
        "Expected Result",
        "Actual Result",
        "Executor Status",
        "Rule Check",
        "Judge Reason",
        "Status",
    ]

    table_data = [
        [
            Paragraph(
                header,
                header_style
            )
            for header in display_headers
        ]
    ]

    for _, row in result.iterrows():

        table_data.append(
            [
                Paragraph(
                    str(row.get("id", "")),
                    small_style
                ),

                Paragraph(
                    str(row.get("risk_level", "")),
                    small_style
                ),

                Paragraph(
                    str(row.get("scenario", "")),
                    small_style
                ),

                Paragraph(
                    str(row.get("expected_result", "")),
                    small_style
                ),

                Paragraph(
                    str(row.get("actual_result", "")),
                    small_style
                ),

                Paragraph(
                    str(row.get("executor_status", "")),
                    small_style
                ),

                Paragraph(
                    str(row.get("rule_check", "")),
                    small_style
                ),

                Paragraph(
                    str(row.get("judge_reason", "")),
                    small_style
                ),

                Paragraph(
                    str(row.get("status", "")),
                    small_style
                ),
            ]
        )

    results_table = Table(
        table_data,
        colWidths=[
            11 * mm,
            17 * mm,
            43 * mm,
            48 * mm,
            53 * mm,
            21 * mm,
            39 * mm,
            40 * mm,
            15 * mm,
        ],
        repeatRows=1,
    )

    results_table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.grey,
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "ALIGN",
                    (8, 1),
                    (8, -1),
                    "CENTER",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    3,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    3,
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    3,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    3,
                ),
            ]
        )
    )

    story.append(results_table)

    story.append(
        Spacer(1, 6 * mm)
    )

    # --------------------------------------------------------
    # EVALUATION NOTE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "<b>Evaluation Note:</b> "
            "Safety scenarios are evaluated using the framework's "
            "deterministic safety rules. No LLM judge is used for "
            "the dedicated safety suite.",
            body_style
        )
    )

    # --------------------------------------------------------
    # BUILD PDF
    # --------------------------------------------------------

    document.build(story)

    return report_pdf


# ============================================================
# MAIN
# ============================================================

async def main():

    cases = pd.read_csv(
        ROOT / "data" / "shared" / "safety_cases.csv"
    ).to_dict("records")

    executor = ChatbotExecutor()

    try:

        await executor._ensure_page()

        result = await SafetyEngine().run(
            cases,
            executor
        )

        # ----------------------------------------------------
        # Save CSV
        # ----------------------------------------------------

        ReportManager().save_safety(
            result
        )

        # ----------------------------------------------------
        # Generate PDF automatically
        # ----------------------------------------------------

        pdf_path = generate_safety_pdf(
            result
        )

        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        print(
            result[
                ["id", "risk_level", "status"]
            ].to_string(
                index=False
            )
        )

        print()
        print(
            "Saved reports/chatbot_app/safety_results.csv"
        )

        print(
            f"Saved {pdf_path}"
        )

        return result

    finally:

        await executor.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())