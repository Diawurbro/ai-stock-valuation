"""
main.py
─────────────────────────────────────────────────────────────────────────────
Orchestration entry point for the AI Stock Valuation Multi-Agent System.

Execution flow (sequential process):
  Task 1 → Interrogator : Generates research questions from a company brief.
  Task 2 → Quant        : Answers questions related to financial metrics.
  Task 3 → Strategist   : Answers questions related to competitive strategy.
  Task 4 → Futurist     : Answers questions related to future S-Curves / tech.
  Task 5 → CIO          : Synthesises Tasks 2-4 → final Buy/Hold/Pass report.

Run:
  python main.py --pdf reports/apple_10k_2023.pdf --company "Apple Inc."

Environment variables required:
  GOOGLE_API_KEY=your_google_api_key_here
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

# Load .env FIRST — must happen before crewai/litellm imports so GOOGLE_API_KEY
# is present in the environment when those modules initialise their HTTP clients.
from dotenv import load_dotenv
load_dotenv()

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from crewai import Crew, Process, Task

# Local modules
from finance_tools import build_yahoo_finance_tool
from agents import StockValuationAgents

# ─────────────────────────────────────────────────────────────────────────────
# Logging configuration
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Task Definitions
# ─────────────────────────────────────────────────────────────────────────────

def build_tasks(agents: dict, company_name: str, ticker: str) -> list[Task]:
    """
    Construct the five sequential CrewAI tasks.

    Key design decisions:
    - `context` parameter: Tasks 2, 3, 4 receive Task 1's output so they know
      *which* questions to focus on. Task 5 receives Tasks 2+3+4 for synthesis.
    - `output_file`: Each task writes its report to disk for auditability.
    - `expected_output` is explicit: CrewAI uses this as a quality check prompt.

    Args:
        agents:       Dict returned by StockValuationAgents.build_all()
        company_name: Human-readable company name (e.g. "Apple Inc.")
        ticker:       Exchange ticker symbol passed to the Quant's YahooFinanceTool
                      (e.g. "AAPL"). Injected directly into the task description.

    Returns:
        List of 5 Tasks in execution order.
    """

    # ── Task 1: Interrogator generates the question framework ─────────────────
    task_interrogate = Task(
        description=(
            f"You are analysing {company_name}. Use the yahoo_finance_data tool (ticker='{ticker}') "
            f"and your general industry knowledge to understand what this company does, "
            f"its size, and its primary markets. "
            f"\n\nThen generate a structured list of 15–20 deep, specific research questions "
            f"that the Quant, Business Strategist, and Futurist agents must answer to build "
            f"a complete investment thesis for {company_name}. "
            f"\n\nOrganise the questions into three labelled sections: "
            f"[FINANCIAL HEALTH], [COMPETITIVE STRATEGY], [FUTURE GROWTH & S-CURVES]. "
            f"Every question must expose a material risk or value driver."
        ),
        expected_output=(
            "A structured markdown document with three clearly labelled sections "
            "([FINANCIAL HEALTH], [COMPETITIVE STRATEGY], [FUTURE GROWTH & S-CURVES]), "
            "each containing 5–7 specific research questions. "
            "Total: 15–20 questions. No generic questions like 'Is the company profitable?' — "
            "every question must name a specific metric, ratio, or strategic factor."
        ),
        agent=agents["interrogator"],
        output_file=f"reports/01_interrogator_questions_{ticker}.md",
    )

    # ── Task 2: Quant answers the [FINANCIAL HEALTH] questions ───────────────
    task_quant = Task(
        description=(
            f"You have received the research question framework for {company_name}. "
            f"Focus exclusively on the [FINANCIAL HEALTH] questions. "
            f"\n\nSTEP 1 — Live data: Call the yahoo_finance_data tool with "
            f"ticker='{ticker}' to fetch the current market price, valuation multiples, "
            f"margins, growth rates, cash flow, and leverage metrics. Record these as your "
            f"live quantitative baseline and label them '[Yahoo Finance]'. "
            f"\n\nSTEP 2 — Analysis: Analyze the metrics and compare them to general industry "
            f"benchmarks and your knowledge of {company_name}'s historical performance. "
            f"\n\nStructure your report with these mandatory sections: "
            f"1. Revenue & Margin Analysis "
            f"2. Free Cash Flow Quality "
            f"3. Balance Sheet & Debt Assessment "
            f"4. Valuation Multiples Benchmarking (live price vs intrinsic value estimate) "
            f"5. Capital Allocation Scorecard "
            f"6. Financial Red Flags (if any) "
            f"\n\nConclude with a one-paragraph Financial Health Rating: "
            f"[Strong / Adequate / Concerning / Critical] with justification."
        ),
        expected_output=(
            "A detailed quantitative financial analysis report (markdown format) with "
            "all 6 mandatory sections completed. Every metric must be labelled with its "
            "source ([Yahoo Finance]). Specific numbers only — "
            "no ranges. Concluding Financial Health Rating paragraph required. "
            "Minimum 600 words. No unsupported assertions."
        ),
        agent=agents["quant"],
        context=[task_interrogate],       # Receives the question list
        output_file=f"reports/02_quant_analysis_{ticker}.md",
    )

    # ── Task 3: Strategist answers the [COMPETITIVE STRATEGY] questions ───────
    task_strategy = Task(
        description=(
            f"You have received the research question framework for {company_name}. "
            f"Focus exclusively on the [COMPETITIVE STRATEGY] questions. "
            f"\n\nUse your general knowledge of the company's industry, products, "
            f"and competitive positioning to answer each question with evidence. "
            f"You may also call the yahoo_finance_data tool if you need live metrics."
            f"\n\nStructure your report with these mandatory sections: "
            f"1. Economic Moat Assessment (type, width, durability evidence) "
            f"2. Competitive Threat Landscape (top 3 threats with evidence) "
            f"3. Management Quality Scorecard (track record, incentives, insider ownership) "
            f"4. Regulatory & ESG Risk Inventory "
            f"5. Business Model Recession Resilience "
            f"\n\nConclude with a Moat Rating: "
            f"[Wide / Narrow / None / Deteriorating] and a one-paragraph justification."
        ),
        expected_output=(
            "A detailed strategic assessment report (markdown format) with all 5 mandatory "
            "sections completed, each claim logically reasoned, and a concluding Moat Rating. "
            "Minimum 600 words."
        ),
        agent=agents["strategist"],
        context=[task_interrogate],
        output_file=f"reports/03_strategy_analysis_{ticker}.md",
    )

    # ── Task 4: Futurist answers the [FUTURE GROWTH & S-CURVES] questions ─────
    task_futurist = Task(
        description=(
            f"You have received the research question framework for {company_name}. "
            f"Focus exclusively on the [FUTURE GROWTH & S-CURVES] questions. "
            f"\n\nUse your general knowledge of the company's technology, product pipeline, "
            f"and industry trends to answer the questions. You can also use the "
            f"yahoo_finance_data tool to get R&D or growth metrics if needed."
            f"\n\nStructure your report with these mandatory sections: "
            f"1. Current S-Curve Position & Evidence "
            f"2. R&D Investment Efficiency Analysis "
            f"3. Next S-Curve Identification (what's the next growth platform?) "
            f"4. TAM Expansion Opportunities (with sizing rationale) "
            f"5. Disruption Risk Assessment (who could make this business obsolete?) "
            f"6. AI / Digital Leverage Score "
            f"\n\nConclude with an overall S-Curve Score from: "
            f"[Early-Stage Growth / Mid-Curve Momentum / Late-Curve Mature / Disruption Risk] "
            f"and explain which companies have successfully navigated a similar S-Curve transition."
        ),
        expected_output=(
            "A forward-looking innovation and S-Curve analysis (markdown format) with all "
            "6 mandatory sections, logical reasoning for growth data, "
            "a concluding S-Curve Score with named comparable companies, and a clear view "
            "on the 5–10 year growth runway. Minimum 600 words."
        ),
        agent=agents["futurist"],
        context=[task_interrogate],
        output_file=f"reports/04_futurist_analysis_{ticker}.md",
    )

    # ── Task 5: CIO synthesises everything → final verdict ────────────────────
    task_cio = Task(
        description=(
            f"You are the final decision-maker. You have received three specialist reports "
            f"for {company_name}: a Quantitative Financial Analysis, a Strategic Assessment, "
            f"and a Futurist S-Curve Analysis. "
            f"\n\nSynthesise these reports into a definitive Executive Investment Summary. "
            f"Do NOT repeat raw data — synthesise insights and resolve any contradictions "
            f"between the three reports. "
            f"\n\nYour output MUST include the following sections in this exact order: "
            f"1. Company Snapshot (3 sentences: what they do, scale, primary market) "
            f"2. Bull Case Investment Thesis (3 paragraphs, evidence-backed) "
            f"3. Bear Case & Key Risks (3 paragraphs, evidence-backed) "
            f"4. Valuation Sanity Check (1 paragraph: is the price fair?) "
            f"5. **FINAL VERDICT**: BUY / HOLD / PASS — Confidence: High/Medium/Low "
            f"   — Primary Driver: [one sentence stating the single most decisive factor] "
            f"6. Suggested Position Sizing & Rationale "
            f"7. 3 Key Metrics to Monitor (with specific thresholds that trigger re-evaluation) "
            f"\n\nThe verdict must be unambiguous. 'It depends' is not an acceptable answer."
        ),
        expected_output=(
            "A professional Executive Investment Summary (markdown format) with all 7 "
            "mandatory sections, a single-word verdict (BUY/HOLD/PASS) in bold with "
            "confidence level, a primary driver sentence, and three specific monitorable "
            "KPIs with quantitative thresholds. Minimum 800 words. Suitable for a "
            "portfolio committee presentation."
        ),
        agent=agents["cio"],
        # CIO synthesises ALL three specialist reports
        context=[task_quant, task_strategy, task_futurist],
        output_file=f"reports/05_cio_investment_summary_{ticker}.md",
    )

    # ── Task 6: Thai Summarizer translates the final verdict ──────────────────
    task_thai_summary = Task(
        description=(
            f"Read the final Executive Investment Summary generated by the CIO "
            f"for {company_name} ({ticker}). Translate and summarize the entire report "
            f"into Thai language. Focus on the core thesis, key risks, valuation, "
            f"and the final BUY/HOLD/PASS verdict."
        ),
        expected_output=(
            "A comprehensive, engaging, and easy-to-read Thai summary of the CIO's "
            "investment report, formatted in Markdown."
        ),
        agent=agents["thai_summarizer"],
        context=[task_cio], # Depends on CIO output
        output_file=f"reports/06_thai_summary_{ticker}.md",
    )

    return [task_interrogate, task_quant, task_strategy, task_futurist, task_cio, task_thai_summary]


# ─────────────────────────────────────────────────────────────────────────────
# Crew Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_crew(
    ticker: str,
    company_name: str | None = None,
) -> Crew:
    """
    Assemble the full multi-agent crew.

    Args:
        company_name:      Human-readable name used in task prompts.
        ticker:            Stock exchange ticker (e.g. "NVDA") for YahooFinanceTool.

    Returns:
        A configured CrewAI Crew ready to kick off.
    """
    # 0. Auto-fetch company name if missing
    if not company_name:
        import yfinance as yf
        try:
            info = yf.Ticker(ticker).info
            company_name = info.get("longName", ticker)
        except Exception:
            company_name = ticker

    # 1. Build the Yahoo Finance tool (no API key needed — uses yfinance)
    logger.info("Initialising Yahoo Finance tool (ticker: %s) …", ticker.upper())
    yahoo_tool = build_yahoo_finance_tool()

    # 2. Instantiate all agents — Quant receives both tools automatically
    logger.info("Building agents …")
    agent_factory = StockValuationAgents(yahoo_tool=yahoo_tool)
    agents = agent_factory.build_all()

    # 3. Build tasks — ticker is injected into the Quant task description
    logger.info("Defining tasks …")
    tasks = build_tasks(agents=agents, company_name=company_name, ticker=ticker)

    # 4. Assemble the Crew
    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,   # Tasks run in strict order: 1→2→3→4→5
        verbose=True,                 # Print agent reasoning to stdout
        memory=False,                 # Stateless between runs (RAG provides context)
    )

    return crew


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Stock Valuation Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --company "Apple Inc." --ticker AAPL
  python main.py --company "Tesla Inc." --ticker TSLA
        """,
    )
    parser.add_argument(
        "--company",
        required=False,
        default="",
        metavar="NAME",
        help='(Optional) Human-readable company name. If omitted, will be fetched automatically.',
    )
    parser.add_argument(
        "--ticker",
        required=True,
        metavar="SYMBOL",
        help='Stock exchange ticker symbol, e.g. "NVDA". Used by the Quant to fetch live data from Yahoo Finance.',
    )
    return parser.parse_args()


def validate_environment() -> None:
    """Fail fast if required environment variables are missing."""
    if not os.environ.get("GOOGLE_API_KEY"):
        logger.error(
            "GOOGLE_API_KEY environment variable is not set. "
            "Export it before running: export GOOGLE_API_KEY=your_key"
        )
        sys.exit(1)


def ensure_report_dir() -> None:
    """Create the reports output directory if it doesn't exist."""
    Path("reports").mkdir(exist_ok=True)


def main() -> None:
    args = parse_args()

    # Pre-flight checks
    validate_environment()
    ensure_report_dir()

    logger.info("=" * 70)
    logger.info("AI STOCK VALUATION SYSTEM — %s", args.ticker.upper())
    logger.info("Started: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 70)

    # Build and run the crew
    crew = build_crew(
        ticker=args.ticker,
        company_name=args.company if args.company else None,
    )

    result = crew.kickoff()

    # ── Final output ──────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 70)
    print("\n" + "═" * 70)
    print("  FINAL CIO INVESTMENT SUMMARY")
    print("═" * 70 + "\n")
    print(result)
    print("\n" + "═" * 70)
    print("  Individual reports saved to: ./reports/")
    print(f"  01_interrogator_questions_{args.ticker}.md")
    print(f"  02_quant_analysis_{args.ticker}.md")
    print(f"  03_strategy_analysis_{args.ticker}.md")
    print(f"  04_futurist_analysis_{args.ticker}.md")
    print(f"  05_cio_investment_summary_{args.ticker}.md")
    print(f"  06_thai_summary_{args.ticker}.md")
    print("═" * 70)

    # ── Auto-upload to Google Drive (if credentials.json exists) ─────────────
    from pathlib import Path as _Path
    uploaded_files: list[dict] = []
    if _Path("credentials.json").exists():
        try:
            from drive_uploader import upload_reports_to_drive
            print("\n📤 Uploading reports to Google Drive …")
            folder_url, uploaded_files = upload_reports_to_drive(ticker=args.ticker)
            if folder_url:
                print(f"✅ Reports uploaded! Open in Drive:\n   {folder_url}")
        except Exception as e:
            logger.warning("Google Drive upload failed: %s", e)
    else:
        print("\n💡 Tip: Add credentials.json to auto-upload reports to Google Drive.")
        print("   See drive_uploader.py for setup instructions.")

    # ── Auto-upload to NotebookLM (if Drive upload succeeded and auth exists) ─
    if uploaded_files:
        import shutil as _shutil
        if _shutil.which("notebooklm") or True:  # notebooklm-py installed in venv
            try:
                from notebooklm_uploader import upload_to_notebooklm
                print("\n📚 Adding reports to NotebookLM …")
                nb_url = upload_to_notebooklm(ticker=args.ticker, uploaded_files=uploaded_files)
                if nb_url:
                    print(f"✅ NotebookLM notebook ready:\n   {nb_url}")
            except Exception as e:
                logger.warning("NotebookLM upload failed: %s", e)
                print("💡 Tip: Run 'notebooklm login' once to enable NotebookLM auto-upload.")


if __name__ == "__main__":
    main()