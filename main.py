"""
main.py
─────────────────────────────────────────────────────────────────────────────
Orchestration entry point for the AI Stock Valuation Multi-Agent System.

Execution flow (sequential process):
  Task 1 → Interrogator    : Generates 20–25 research questions from a company brief.
  Task 2 → Quant           : Answers [FINANCIAL HEALTH] questions.
  Task 3 → Strategist      : Answers [COMPETITIVE STRATEGY] questions.
  Task 4 → Futurist        : Answers [FUTURE GROWTH & S-CURVES] questions.
  Task 5 → Risk Analyst    : Macro, geopolitical, and sector risk analysis.  [NEW]
  Task 6 → ESG Auditor     : ESG & corporate governance deep-dive.            [NEW]
  Task 7 → Devil’s Advocate: Bear case & valuation stress-test.               [NEW]
  Task 8 → CIO             : Synthesises Tasks 2–7 → final Buy/Hold/Pass report.
  Task 9 → Thai Summarizer : Translates verdict into Thai.

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
    Construct the nine sequential CrewAI tasks.

    Key design decisions:
    - `context` parameter: Tasks 2–7 receive Task 1’s output (the question list).
      Task 8 (CIO) receives Tasks 2–7 for full-spectrum synthesis.
    - `output_file`: Each task writes its report to disk for auditability.
    - `expected_output` is explicit: CrewAI uses this as a quality check prompt.

    Args:
        agents:       Dict returned by StockValuationAgents.build_all()
        company_name: Human-readable company name (e.g. "Apple Inc.")
        ticker:       Exchange ticker symbol passed to the Quant’s YahooFinanceTool
                      (e.g. "AAPL"). Injected directly into the task description.

    Returns:
        List of 9 Tasks in execution order.
    """

    # ── Task 1: Interrogator generates the question framework ─────────────────
    task_interrogate = Task(
        description=(
            f"You are analysing {company_name}. Use the yahoo_finance_data tool (ticker='{ticker}') "
            f"and your general industry knowledge to understand what this company does, "
            f"its size, and its primary markets. "
            f"\n\nThen generate a structured list of 20–25 deep, specific research questions "
            f"that the Quant, Business Strategist, Futurist, Risk Analyst, ESG Auditor, and "
            f"Devil's Advocate agents must answer to build a complete investment thesis for {company_name}. "
            f"\n\nOrganise the questions into five labelled sections: "
            f"[FINANCIAL HEALTH], [COMPETITIVE STRATEGY], [FUTURE GROWTH & S-CURVES], "
            f"[MACRO & GEOPOLITICAL RISKS], [ESG & GOVERNANCE]. "
            f"Every question must expose a material risk or value driver."
        ),
        expected_output=(
            "A structured markdown document with five clearly labelled sections "
            "([FINANCIAL HEALTH], [COMPETITIVE STRATEGY], [FUTURE GROWTH & S-CURVES], "
            "[MACRO & GEOPOLITICAL RISKS], [ESG & GOVERNANCE]), "
            "each containing 4–5 specific research questions. "
            "Total: 20–25 questions. No generic questions — "
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

    # ── Task 5: Risk Analyst — Macro & Geopolitical Risk ─────────────────────
    task_risk = Task(
        description=(
            f"You have received the research question framework for {company_name}. "
            f"Focus on the [MACRO & GEOPOLITICAL RISKS] questions. "
            f"\n\nUse the yahoo_finance_data tool (ticker='{ticker}') for any live "
            f"financial metrics that support your risk assessment (e.g. debt levels, "
            f"revenue breakdown, current valuation as a base for stress tests). "
            f"\n\nStructure your report with these mandatory sections: "
            f"1. Interest Rate Sensitivity Analysis (+200bps shock scenario) "
            f"2. Geopolitical & Supply Chain Risk Map (key country exposures, tariffs) "
            f"3. Sector Cycle Positioning (defensive vs cyclical, where in the cycle?) "
            f"4. Currency & FX Exposure Assessment "
            f"5. Regulatory & Policy Risk Inventory "
            f"6. Black Swan Scenarios (2 extreme downside cases with probability estimates) "
            f"\n\nConclude with an overall Macro Risk Rating: "
            f"[Low / Moderate / High / Extreme] with one-paragraph justification."
        ),
        expected_output=(
            "A detailed macro and sector risk report (markdown format) with all 6 mandatory "
            "sections, specific probability-weighted scenarios, and a concluding Macro Risk "
            "Rating with justification. Minimum 600 words. All claims backed by data or "
            "logical reasoning from publicly available information."
        ),
        agent=agents["risk_analyst"],
        context=[task_interrogate],
        output_file=f"reports/05_risk_analysis_{ticker}.md",
    )

    # ── Task 6: ESG Auditor — Governance & ESG Deep-Dive ─────────────────────
    task_esg = Task(
        description=(
            f"You have received the research question framework for {company_name}. "
            f"Focus on the [ESG & GOVERNANCE] questions. "
            f"\n\nUse the yahoo_finance_data tool (ticker='{ticker}') to retrieve any "
            f"available governance or financial metrics. Supplement with your general "
            f"knowledge of the company's governance structure and ESG record. "
            f"\n\nStructure your report with these mandatory sections: "
            f"1. Board Quality Assessment (independence, diversity, expertise, tenure) "
            f"2. Executive Compensation Analysis (alignment with shareholder value, ROIC link) "
            f"3. Shareholder Rights & Anti-Takeover Provisions "
            f"4. Environmental Commitments & Carbon Risk (Scope 1/2/3, net-zero credibility) "
            f"5. Social & Supply Chain Ethics (labour, data privacy, community impact) "
            f"6. Governance Red Flags & Accounting Integrity "
            f"\n\nAssign an ESG Risk Score [Low / Medium / High / Critical] per pillar "
            f"and an overall score with justification."
        ),
        expected_output=(
            "A comprehensive ESG and governance audit report (markdown format) with all 6 "
            "mandatory sections, individual pillar risk scores, an overall ESG Risk Score "
            "with justification, and any identified red flags highlighted. Minimum 600 words."
        ),
        agent=agents["esg_auditor"],
        context=[task_interrogate],
        output_file=f"reports/06_esg_governance_{ticker}.md",
    )

    # ── Task 7: Devil's Advocate — Bear Case & Stress Test ───────────────────
    task_devils = Task(
        description=(
            f"Your mission is to stress-test the investment case for {company_name} "
            f"and construct the most rigorous bear case possible. "
            f"\n\nUse the yahoo_finance_data tool (ticker='{ticker}') to get the current "
            f"valuation metrics (P/E, EV/EBITDA, P/FCF, price) as the baseline for your "
            f"stress-test models. "
            f"\n\nStructure your report with these mandatory sections: "
            f"1. Valuation Stress Test: Base / Bear / Deep Bear scenarios with implied price targets "
            f"2. Dangerous Consensus Beliefs (what is the Street most wrong about?) "
            f"3. Hidden Liabilities & Off-Balance-Sheet Risks "
            f"4. Accounting Quality & Earnings Integrity (accruals ratio analysis) "
            f"5. Specific Moat Erosion Scenario (name the competitor or technology) "
            f"6. Management Execution Risk (cite 2 specific prior failures) "
            f"\n\nConclude with a Margin of Safety Assessment: at the current price, "
            f"how much downside protection exists if the bear case materialises?"
        ),
        expected_output=(
            "A rigorous bear case and valuation stress-test report (markdown format) with "
            "all 6 mandatory sections, 3 explicitly modelled price scenarios with stated "
            "assumptions, specific named threats, and a Margin of Safety conclusion. "
            "Minimum 700 words. This report must challenge — not confirm — the bull case."
        ),
        agent=agents["devils_advocate"],
        context=[task_quant, task_strategy, task_futurist],
        output_file=f"reports/07_devils_advocate_{ticker}.md",
    )

    # ── Task 8: CIO synthesises everything → final verdict ───────────────────
    task_cio = Task(
        description=(
            f"You are the final decision-maker. You have received SIX specialist reports "
            f"for {company_name}: Quantitative Financial Analysis, Strategic Assessment, "
            f"Futurist S-Curve Analysis, Macro & Geopolitical Risk Report, ESG & Governance "
            f"Audit, and a Devil's Advocate Bear Case. "
            f"\n\nSynthesise ALL six reports into a definitive Executive Investment Summary. "
            f"Do NOT repeat raw data — synthesise insights, resolve contradictions between "
            f"reports, and weigh the bull case against the bear case rigorously. "
            f"\n\nYour output MUST include the following sections in this exact order: "
            f"1. Company Snapshot (3 sentences: what they do, scale, primary market) "
            f"2. Bull Case Investment Thesis (3 paragraphs, evidence-backed) "
            f"3. Bear Case & Key Risks (3 paragraphs, referencing the Devil's Advocate report) "
            f"4. Macro & ESG Risk Overlay (1 paragraph integrating risk + governance findings) "
            f"5. Valuation Sanity Check (1 paragraph: is the price fair given all evidence?) "
            f"6. **FINAL VERDICT**: BUY / HOLD / PASS — Confidence: High/Medium/Low "
            f"   — Primary Driver: [one sentence stating the single most decisive factor] "
            f"7. Suggested Position Sizing & Rationale "
            f"8. 3 Key Metrics to Monitor (with specific thresholds that trigger re-evaluation) "
            f"\n\nThe verdict must be unambiguous. 'It depends' is not acceptable."
        ),
        expected_output=(
            "A professional Executive Investment Summary (markdown format) with all 8 "
            "mandatory sections, a single-word verdict (BUY/HOLD/PASS) in bold with "
            "confidence level, a primary driver sentence, and three specific monitorable "
            "KPIs with quantitative thresholds. Minimum 1000 words. Suitable for a "
            "portfolio committee presentation."
        ),
        agent=agents["cio"],
        context=[task_quant, task_strategy, task_futurist, task_risk, task_esg, task_devils],
        output_file=f"reports/08_cio_investment_summary_{ticker}.md",
    )

    # ── Task 9: Thai Summarizer translates the final verdict ─────────────────
    task_thai_summary = Task(
        description=(
            f"Read the final Executive Investment Summary generated by the CIO "
            f"for {company_name} ({ticker}). Translate and summarize the entire report "
            f"into Thai language. Cover all 8 sections: company snapshot, bull case, "
            f"bear case, macro/ESG overlay, valuation check, final verdict, position "
            f"sizing, and KPIs to monitor."
        ),
        expected_output=(
            "A comprehensive, engaging, and easy-to-read Thai summary of the CIO's "
            "investment report covering all key sections, formatted in Markdown. "
            "Minimum 400 words in Thai."
        ),
        agent=agents["thai_summarizer"],
        context=[task_cio],
        output_file=f"reports/09_thai_summary_{ticker}.md",
    )

    return [
        task_interrogate, task_quant, task_strategy, task_futurist,
        task_risk, task_esg, task_devils, task_cio, task_thai_summary,
    ]


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
    print(f"  05_risk_analysis_{args.ticker}.md")
    print(f"  06_esg_governance_{args.ticker}.md")
    print(f"  07_devils_advocate_{args.ticker}.md")
    print(f"  08_cio_investment_summary_{args.ticker}.md")
    print(f"  09_thai_summary_{args.ticker}.md")
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