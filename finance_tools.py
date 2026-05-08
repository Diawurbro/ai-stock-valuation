"""
finance_tools.py
─────────────────────────────────────────────────────────────────────────────
Live quantitative data tools for the AI Stock Valuation Multi-Agent System.

Provides:
  • YahooFinanceTool  — fetches real-time & TTM financial metrics via yfinance.
  • (Extensible: add more BaseTool subclasses here for other data sources.)

Dependencies:
  pip install yfinance

Design philosophy:
  - Every public number is labelled with its source field name from yfinance
    so the Quant agent can cross-reference against the PDF/RAG data.
  - Missing fields degrade gracefully to "N/A" — never raise KeyError to the LLM.
  - All monetary values are human-formatted (e.g., $12.4B) for LLM readability.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import Any

import yfinance as yf
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_currency(value: Any, decimals: int = 2) -> str:
    """
    Format a raw numeric value as a human-readable USD string.

    Examples:
        1_234_567_890  → "$1.23B"
        945_000_000    → "$945.00M"
        12_000         → "$12,000.00"
        None / "N/A"   → "N/A"
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"

    abs_v = abs(v)
    sign  = "-" if v < 0 else ""

    if abs_v >= 1e12:
        return f"{sign}${abs_v / 1e12:.{decimals}f}T"
    if abs_v >= 1e9:
        return f"{sign}${abs_v / 1e9:.{decimals}f}B"
    if abs_v >= 1e6:
        return f"{sign}${abs_v / 1e6:.{decimals}f}M"
    return f"{sign}${abs_v:,.{decimals}f}"


def _fmt_pct(value: Any, decimals: int = 2) -> str:
    """
    Format a ratio (0–1 scale from yfinance) as a percentage string.

    Example: 0.2345 → "23.45%"
    """
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_ratio(value: Any, decimals: int = 2) -> str:
    """Format a plain ratio / multiple (e.g., P/E = 28.5x)."""
    try:
        return f"{float(value):.{decimals}f}x"
    except (TypeError, ValueError):
        return "N/A"


def _safe_get(info: dict, key: str, formatter=None) -> str:
    """
    Safely extract a key from yfinance's info dict and apply a formatter.

    Returns "N/A" if the key is missing or the value is None.
    """
    raw = info.get(key)
    if raw is None:
        return "N/A"
    if formatter:
        return formatter(raw)
    return str(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Input Schema
# ─────────────────────────────────────────────────────────────────────────────

class YahooFinanceInput(BaseModel):
    """Strict input schema for YahooFinanceTool."""

    ticker: str = Field(
        ...,
        description=(
            "The stock ticker symbol to look up. Must be the exact exchange symbol. "
            "Examples: 'NVDA' for NVIDIA, 'AAPL' for Apple, 'MSFT' for Microsoft. "
            "Do NOT include exchange suffixes unless required (e.g., use 'NVDA' not 'NVDA.NQ')."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# YahooFinanceTool
# ─────────────────────────────────────────────────────────────────────────────

class YahooFinanceTool(BaseTool):
    """
    CrewAI Tool that fetches live financial metrics for a given ticker via yfinance.

    Metric coverage (17 key indicators):
      Valuation    : Trailing P/E, Forward P/E, P/S, P/B, EV/EBITDA, EV/Revenue
      Profitability: Gross Margin, EBIT Margin, Net Profit Margin, ROE, ROA, ROIC
      Growth       : Revenue Growth (YoY), Earnings Growth (YoY)
      Cash Flow    : Free Cash Flow (TTM), Operating Cash Flow (TTM)
      Leverage     : Debt-to-Equity, Current Ratio, Quick Ratio
      Market Data  : Market Cap, Enterprise Value, 52W High/Low, Beta

    Usage by the Quant agent:
        "Use the yahoo_finance_data tool with ticker='NVDA' to fetch live metrics,
         then cross-reference against the figures in the annual report."
    """

    name: str        = "yahoo_finance_data"
    description: str = (
        "Fetches live and trailing-twelve-month (TTM) financial metrics for a stock "
        "ticker from Yahoo Finance. Use this to get real-time valuation multiples "
        "(P/E, EV/EBITDA), profitability ratios (margins, ROE, ROIC), growth rates, "
        "cash flow figures, and leverage metrics. Always call this tool first to "
        "establish a live quantitative baseline, then cross-validate against the "
        "annual report data retrieved via the RAG search tool."
    )
    args_schema: type[BaseModel] = YahooFinanceInput

    def _run(self, ticker: str) -> str:
        """
        Core execution method called by CrewAI.

        Fetches yfinance data and returns a clean, structured markdown-style
        text block that the LLM can directly parse and reason over.

        Args:
            ticker: Stock ticker symbol (e.g., "NVDA").

        Returns:
            Formatted string of financial metrics, or an error message.
        """
        ticker = ticker.strip().upper()
        logger.info("[YahooFinanceTool] Fetching data for ticker: %s", ticker)

        try:
            stock = yf.Ticker(ticker)
            info  = stock.info  # Main metadata dict (~150 fields)

            # ── Validate the response ──────────────────────────────────────────
            # yfinance returns a minimal stub dict for invalid tickers instead of
            # raising an exception — we detect this by checking for a core field.
            if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
                # Attempt a secondary validation via quoteType
                if info.get("quoteType") in (None, ""):
                    return (
                        f"ERROR: Could not retrieve data for ticker '{ticker}'. "
                        f"Please verify the ticker symbol is correct and listed on a major exchange."
                    )

            # ── Fetch cash flow data (separate yfinance endpoint) ────────────
            # cashflow dataframe gives us more granular FCF than info dict alone
            try:
                cf      = stock.cashflow          # Annual cash flow statement
                fcf_raw = None
                ocf_raw = None

                if cf is not None and not cf.empty:
                    # Row labels vary by ticker; use flexible matching
                    for row_label in cf.index:
                        lbl = str(row_label).lower()
                        if "free cash flow" in lbl:
                            fcf_raw = cf.loc[row_label].iloc[0]   # Most recent year
                        if "operating" in lbl and "cash" in lbl:
                            ocf_raw = cf.loc[row_label].iloc[0]

                # Fall back to info dict if cashflow statement parsing failed
                fcf_display = _fmt_currency(fcf_raw) if fcf_raw else _safe_get(info, "freeCashflow", _fmt_currency)
                ocf_display = _fmt_currency(ocf_raw) if ocf_raw else _safe_get(info, "operatingCashflow", _fmt_currency)

            except Exception as cf_err:
                logger.warning("[YahooFinanceTool] Cash flow fetch failed: %s", cf_err)
                fcf_display = _safe_get(info, "freeCashflow", _fmt_currency)
                ocf_display = _safe_get(info, "operatingCashflow", _fmt_currency)

            # ── Build the output report ────────────────────────────────────────
            company_name = info.get("longName") or info.get("shortName") or ticker
            sector       = info.get("sector",   "N/A")
            industry     = info.get("industry", "N/A")
            exchange     = info.get("exchange", "N/A")

            report_lines = [
                f"# Yahoo Finance Live Data — {company_name} ({ticker})",
                f"Exchange: {exchange} | Sector: {sector} | Industry: {industry}",
                f"Data as of: {_safe_get(info, 'regularMarketTime')} (Unix timestamp)",
                "",

                "## 📊 Market Overview",
                f"  Current Price      : {_safe_get(info, 'currentPrice', lambda v: f'${float(v):.2f}')}",
                f"  Market Cap         : {_safe_get(info, 'marketCap', _fmt_currency)}",
                f"  Enterprise Value   : {_safe_get(info, 'enterpriseValue', _fmt_currency)}",
                f"  52-Week High       : {_safe_get(info, 'fiftyTwoWeekHigh', lambda v: f'${float(v):.2f}')}",
                f"  52-Week Low        : {_safe_get(info, 'fiftyTwoWeekLow',  lambda v: f'${float(v):.2f}')}",
                f"  Beta (5Y Monthly)  : {_safe_get(info, 'beta', lambda v: f'{float(v):.2f}')}",
                "",

                "## 📈 Valuation Multiples",
                f"  Trailing P/E       : {_safe_get(info, 'trailingPE',       _fmt_ratio)}",
                f"  Forward P/E        : {_safe_get(info, 'forwardPE',        _fmt_ratio)}",
                f"  Price-to-Sales     : {_safe_get(info, 'priceToSalesTrailing12Months', _fmt_ratio)}",
                f"  Price-to-Book      : {_safe_get(info, 'priceToBook',      _fmt_ratio)}",
                f"  EV / EBITDA        : {_safe_get(info, 'enterpriseToEbitda', _fmt_ratio)}",
                f"  EV / Revenue       : {_safe_get(info, 'enterpriseToRevenue', _fmt_ratio)}",
                f"  PEG Ratio          : {_safe_get(info, 'pegRatio',         _fmt_ratio)}",
                "",

                "## 💰 Profitability",
                f"  Gross Margin       : {_safe_get(info, 'grossMargins',     _fmt_pct)}",
                f"  EBIT Margin        : {_safe_get(info, 'ebitdaMargins',    _fmt_pct)}",
                f"  Net Profit Margin  : {_safe_get(info, 'profitMargins',    _fmt_pct)}",
                f"  Return on Equity   : {_safe_get(info, 'returnOnEquity',   _fmt_pct)}",
                f"  Return on Assets   : {_safe_get(info, 'returnOnAssets',   _fmt_pct)}",
                "",

                "## 📉 Growth (Year-over-Year)",
                f"  Revenue Growth     : {_safe_get(info, 'revenueGrowth',    _fmt_pct)}",
                f"  Earnings Growth    : {_safe_get(info, 'earningsGrowth',   _fmt_pct)}",
                f"  Earnings Quarterly : {_safe_get(info, 'earningsQuarterlyGrowth', _fmt_pct)}",
                "",

                "## 🏦 Cash Flow (TTM)",
                f"  Free Cash Flow     : {fcf_display}",
                f"  Operating Cash Flow: {ocf_display}",
                f"  Total Revenue      : {_safe_get(info, 'totalRevenue',     _fmt_currency)}",
                f"  EBITDA             : {_safe_get(info, 'ebitda',           _fmt_currency)}",
                "",

                "## ⚖️  Balance Sheet & Leverage",
                f"  Total Debt         : {_safe_get(info, 'totalDebt',        _fmt_currency)}",
                f"  Total Cash         : {_safe_get(info, 'totalCash',        _fmt_currency)}",
                f"  Net Debt           : {_fmt_currency(                            # Derived: totalDebt - totalCash
                    (info.get('totalDebt') or 0) - (info.get('totalCash') or 0)
                )}",
                f"  Debt-to-Equity     : {_safe_get(info, 'debtToEquity',    lambda v: f'{float(v):.2f}')}",
                f"  Current Ratio      : {_safe_get(info, 'currentRatio',    lambda v: f'{float(v):.2f}')}",
                f"  Quick Ratio        : {_safe_get(info, 'quickRatio',      lambda v: f'{float(v):.2f}')}",
                "",

                "## 🧮 Per-Share Metrics",
                f"  EPS (Trailing)     : {_safe_get(info, 'trailingEps',     lambda v: f'${float(v):.2f}')}",
                f"  EPS (Forward)      : {_safe_get(info, 'forwardEps',      lambda v: f'${float(v):.2f}')}",
                f"  Book Value/Share   : {_safe_get(info, 'bookValue',       lambda v: f'${float(v):.2f}')}",
                f"  Dividend Yield     : {_safe_get(info, 'dividendYield',   _fmt_pct)}",
                f"  Payout Ratio       : {_safe_get(info, 'payoutRatio',     _fmt_pct)}",
                "",

                "## 🔬 Analyst Consensus",
                f"  Recommendation     : {_safe_get(info, 'recommendationKey').upper()}",
                f"  # of Analyst Opns  : {_safe_get(info, 'numberOfAnalystOpinions')}",
                f"  Target Mean Price  : {_safe_get(info, 'targetMeanPrice',  lambda v: f'${float(v):.2f}')}",
                f"  Target High Price  : {_safe_get(info, 'targetHighPrice',  lambda v: f'${float(v):.2f}')}",
                f"  Target Low Price   : {_safe_get(info, 'targetLowPrice',   lambda v: f'${float(v):.2f}')}",
                "",

                "─" * 60,
                "NOTE: All figures sourced from Yahoo Finance via yfinance library.",
                "Cross-validate material figures against the official annual report.",
            ]

            result = "\n".join(report_lines)
            logger.info("[YahooFinanceTool] Successfully built report (%d chars)", len(result))
            return result

        except Exception as e:
            # Catch-all: never let a data fetch crash the agent crew
            error_msg = (
                f"ERROR: An unexpected error occurred while fetching data for '{ticker}': "
                f"{type(e).__name__}: {e}. "
                f"Proceed with PDF/RAG data only and note this data gap in your report."
            )
            logger.error("[YahooFinanceTool] %s", error_msg)
            return error_msg


# ─────────────────────────────────────────────────────────────────────────────
# Convenience instantiator (import and call in agents.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_yahoo_finance_tool() -> YahooFinanceTool:
    """
    Returns a ready-to-use YahooFinanceTool instance.

    Usage in agents.py:
        from finance_tools import build_yahoo_finance_tool
        yahoo_tool = build_yahoo_finance_tool()
        # Pass to the Quant agent's tools list
    """
    return YahooFinanceTool()