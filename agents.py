"""
agents.py
─────────────────────────────────────────────────────────────────────────────
Defines all 5 CrewAI Agents for the AI Stock Valuation Multi-Agent System.

Agent Roster:
  1. The Interrogator   — Generates deep investigative research questions.
  2. The Quant          — Financial health, cash flow, valuation metrics.
  3. The Business Strategist — Moats, risks, management quality.
  4. The Futurist       — R&D, technology trends, S-Curve analysis.
  5. The CIO            — Synthesises all reports → Buy / Hold / Pass.
  6. The Thai Summarizer — Translates the final verdict into Thai.

Each agent receives the RAG tool so all factual claims are grounded in the
official financial documents ingested during the RAG setup phase.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from crewai import Agent, LLM
from crewai.tools import BaseTool

# Local — live market data tool (hybrid RAG + live data for The Quant)
from finance_tools import YahooFinanceTool, build_yahoo_finance_tool


# ─────────────────────────────────────────────────────────────────────────────
# LLM Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_llm(temperature: float = 0.1) -> LLM:
    """
    Build the shared Gemini 2.0 Flash LLM instance.

    Low temperature (0.1) keeps financial analysis deterministic and fact-
    grounded. The Futurist uses a slightly higher temp for creative reasoning.

    Requires the GOOGLE_API_KEY environment variable to be set.
    """
    import os
    return LLM(
        model="gemini/gemini-2.5-flash-lite",
        temperature=temperature,
        max_tokens=8_192,
        api_key=os.environ.get("GOOGLE_API_KEY"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent Factory Class
# ─────────────────────────────────────────────────────────────────────────────

class StockValuationAgents:
    """
    Factory that builds and holds all 5 specialised CrewAI agents.

    Usage:
        rag_tool   = build_rag_tool(["path/to/report.pdf"])
        yahoo_tool = build_yahoo_finance_tool()          # NEW: live market data
        factory    = StockValuationAgents(rag_tool=rag_tool, yahoo_tool=yahoo_tool)
        agents     = factory.build_all()
        # agents["interrogator"], agents["quant"], …
    """

    def __init__(self, yahoo_tool: YahooFinanceTool | None = None):
        """
        Args:
            yahoo_tool: A YahooFinanceTool instance (from finance_tools.py).
                        Optional — defaults to a fresh instance if not supplied.
        """
        # If caller does not pass a pre-built yahoo_tool, create one automatically.
        self._yahoo_tool = yahoo_tool if yahoo_tool is not None else build_yahoo_finance_tool()
        self._llm_analytical = build_llm(temperature=0.1)   # strict / factual
        self._llm_creative    = build_llm(temperature=0.3)   # slightly looser for strategy/future

    # ── Individual agent builders ─────────────────────────────────────────────

    def build_interrogator(self) -> Agent:
        """
        Agent 1 — The Interrogator

        Role: Reads a company overview and then generates a structured list of
        the sharpest, most financially relevant questions that the downstream
        specialist agents must answer from the source documents.

        Why it matters: Forces agents to address *specific* data points rather
        than producing generic commentary. Acts as the system's "audit checklist."
        """
        return Agent(
            role="Chief Investment Research Interrogator",
            goal=(
                "Generate a definitive, structured list of 15–20 targeted research "
                "questions covering financial health, competitive positioning, management "
                "quality, and future growth vectors. Every question must expose either a risk or a "
                "value driver that materially affects a Buy/Hold/Pass decision."
            ),
            backstory=(
                "You are a former Goldman Sachs equity research director with 20 years "
                "of experience stress-testing investment theses. You have a razor-sharp "
                "instinct for identifying the 'killer question' — the one data point that "
                "either confirms or destroys an investment case. You are deeply sceptical "
                "by nature and believe that the best investment questions are the ones "
                "management least wants to answer. You think in frameworks: DuPont, Porter's "
                "Five Forces, and the Buffett Owner's Manual are your native languages."
            ),
            tools=[self._yahoo_tool],
            llm=self._llm_analytical,
            verbose=True,
            allow_delegation=False,    # The Interrogator works alone
            max_iter=3,                # Prevent infinite tool-call loops
        )

    def build_quant(self) -> Agent:
        """
        Agent 2 — The Quant

        Role: Pure quantitative analysis of financials. Extracts, calculates, and
        interprets every material financial metric: revenue growth, margins, FCF,
        debt load, ROIC, valuation multiples, and capital allocation patterns.
        """
        return Agent(
            role="Senior Quantitative Financial Analyst",
            goal=(
                "Produce a rigorous, numbers-first analysis of the company's financial "
                "health. WORKFLOW: (1) ALWAYS call the yahoo_finance_data tool first with "
                "the company's ticker to fetch live market prices, valuation multiples, and "
                "TTM ratios as your quantitative baseline. (2) Analyze these metrics logically "
                "to extract insights about the company's current financial standing. "
                "Coverage: (a) Valuation Multiples (b) Profitability (c) Growth (d) Cash Flow "
                "(e) Leverage. "
                "Flag red flags (margin compression, FCF-earnings divergence, rising leverage) "
                "with explicit source citations from Yahoo Finance."
            ),
            backstory=(
                "You are a CFA charterholder and ex-quant at Citadel Securities who spent "
                "a decade building financial models in Python and R. You are allergic to "
                "vague language: every claim must be backed by a number, a ratio, or a "
                "direct quote from the filing. You know that 80% of investment mistakes "
                "come from misreading cash flow statements, and you never confuse GAAP "
                "earnings with economic earnings. You distrust management guidance unless "
                "it is corroborated by three years of consistent execution."
            ),
            tools=[self._yahoo_tool],
            llm=self._llm_analytical,
            verbose=True,
            allow_delegation=False,
            max_iter=6,   # +1 iteration budget for the extra yahoo tool call
        )

    def build_business_strategist(self) -> Agent:
        """
        Agent 3 — The Business Strategist

        Role: Qualitative / strategic layer. Evaluates the durability of the
        competitive moat, the quality of management, the risk landscape, and
        the clarity of the company's capital allocation philosophy.
        """
        return Agent(
            role="Senior Business Strategy & Competitive Intelligence Analyst",
            goal=(
                "Deliver a comprehensive strategic assessment covering: "
                "(a) Economic Moat — type (network effect, cost advantage, switching cost, "
                "intangible asset), width, and evidence of durability; "
                "(b) Competitive Risks — who poses the greatest 3–5 year threat and why; "
                "(c) Management Quality — track record on guidance accuracy, ROIC vs cost "
                "of capital, shareholder-friendliness, and skin-in-the-game (insider ownership); "
                "Regulatory & ESG Risks — material headwinds from policy or litigation; "
                "(e) Business Model Resilience — how does this company perform in a recession "
                "or a 200bps interest rate shock? "
                "Base conclusions on your general knowledge of the industry and company, "
                "and cross-reference with live quantitative data."
            ),
            backstory=(
                "You are a former McKinsey Partner and strategy consultant who pivoted into "
                "buy-side equity research. You have advised Fortune 500 boards on competitive "
                "strategy and have a practitioner's understanding of how moats are built — "
                "and destroyed. You have lived through dot-com, the GFC, and COVID disruptions "
                "and have internalised that narrative quality ('great story') without financial "
                "discipline is the most dangerous investment trap. You use Morningstar's moat "
                "framework and Phil Fisher's scuttlebutt methodology in every analysis."
            ),
            tools=[self._yahoo_tool],
            llm=self._llm_creative,
            verbose=True,
            allow_delegation=False,
            max_iter=4,
        )

    def build_futurist(self) -> Agent:
        """
        Agent 4 — The Futurist

        Role: Forward-looking analysis. Examines the company's R&D pipeline, its
        positioning on technology S-Curves, TAM expansion opportunities, and whether
        management is allocating capital toward the right future growth vectors.
        """
        return Agent(
            role="Technology & Innovation Futurist / S-Curve Analyst",
            goal=(
                "Produce a forward-looking growth analysis covering: "
                "(a) S-Curve Positioning — identify which S-Curves the company currently "
                "rides (mature, growth, or early) and whether it is investing to jump to "
                "the next one before disruption hits; "
                "(b) R&D Efficiency — R&D trends and any breakthrough products; "
                "(c) TAM Expansion — credible addressable market sizing for new initiatives; "
                "(d) Disruption Risk — is the core business on a declining S-Curve? "
                "Who could make this company irrelevant in 7–10 years? "
                "(e) AI / Digital / Platform Leverage — how is the company monetising "
                "technology to achieve non-linear scale? "
                "Assign a qualitative S-Curve score: [Early-Stage Growth / Mid-Curve Momentum / "
                "Late-Curve Mature / Disruption Risk] with justification."
            ),
            backstory=(
                "You are a Silicon Valley technologist turned investor, formerly Head of "
                "Emerging Technology at Andreessen Horowitz. You have deeply studied the "
                "S-Curve theory (Richard Foster), the Innovator's Dilemma (Christensen), "
                "and Exponential Organizations (Ismail). You believe that 90% of long-term "
                "alpha comes from correctly identifying companies at the inflection point "
                "of a new S-Curve before consensus recognises it. You are equally focused "
                "on avoiding value traps: companies that look cheap because the market "
                "correctly sees them as buggy-whip makers in a Tesla world."
            ),
            tools=[self._yahoo_tool],
            llm=self._llm_creative,
            verbose=True,
            allow_delegation=False,
            max_iter=4,
        )

    def build_cio(self) -> Agent:
        """
        Agent 5 — The CIO (Chief Investment Officer)

        Role: Synthesis engine. Reads the Quant, Strategist, and Futurist reports,
        weighs them against each other, and delivers a final, unambiguous investment
        verdict with clear conviction and reasoning.

        The CIO does NOT use the RAG tool directly — it synthesises the outputs of
        the three specialist agents, trusting them to have cited the source documents.
        """
        return Agent(
            role="Chief Investment Officer & Portfolio Decision Maker",
            goal=(
                "Read the quantitative analysis, strategic assessment, and futurist report, "
                "then produce a definitive Executive Investment Summary that includes: "
                "(a) A 3-paragraph Investment Thesis stating the core bull case; "
                "(b) A 3-paragraph Bear Case / Key Risks section; "
                "(c) A Valuation Sanity Check — is the current price consistent with "
                "intrinsic value given the evidence? "
                "(d) A clear, unambiguous verdict: BUY / HOLD / PASS with a confidence "
                "level (High / Medium / Low) and the single most important factor driving "
                "the decision; "
                "(e) Suggested position sizing rationale (e.g., 'Core position 5% of "
                "portfolio' or 'Speculative <1%'); "
                "(f) 3 Key Metrics to Monitor over the next 12 months as leading indicators "
                "that would cause a rating change."
            ),
            backstory=(
                "You are a legendary Chief Investment Officer modelled on the discipline of "
                "Howard Marks (Oaktree) and the analytical rigour of Terry Smith (Fundsmith). "
                "You have managed $10B+ portfolios through multiple market cycles. You never "
                "issue a BUY without understanding the bear case, and you never issue a PASS "
                "without acknowledging what the market may be missing. You believe the purpose "
                "of analysis is not to sound smart — it is to make one high-conviction decision "
                "that can be clearly communicated to a board in under two minutes. You are "
                "the final decision-maker and you own the verdict completely."
            ),
            tools=[],             # CIO synthesises — no RAG needed at this stage
            llm=self._llm_analytical,
            verbose=True,
            allow_delegation=False,
            max_iter=3,
        )

    def build_thai_summarizer(self) -> Agent:
        """
        Agent 6 — The Thai Financial Communicator & Summarizer

        Role: Takes the complex English financial reports (especially the CIO's verdict)
        and translates/summarizes them into engaging, easy-to-understand Thai language.
        """
        return Agent(
            role="Thai Financial Communicator & Summarizer",
            goal=(
                "Take the final Executive Investment Summary and translate/summarize it "
                "into natural, professional, and easy-to-understand Thai language. "
                "Ensure that financial jargon is explained or translated accurately. "
                "Your output must be highly engaging and formatted nicely using Markdown. "
                "It should be suitable for retail investors in Thailand."
            ),
            backstory=(
                "You are an expert bilingual financial journalist and analyst based in Bangkok. "
                "You have a talent for breaking down complex Wall Street jargon and deep "
                "quantitative analysis into digestible, engaging Thai summaries. You understand "
                "that retail investors need clear takeaways: what is the company doing, what are "
                "the risks, and what is the final verdict."
            ),
            tools=[],
            llm=self._llm_creative,
            verbose=True,
            allow_delegation=False,
            max_iter=3,
        )

    # ── Build All ──────────────────────────────────────────────────────────────

    def build_all(self) -> dict[str, Agent]:
        """
        Instantiate and return all 5 agents as a named dictionary.

        Returns:
            {
                "interrogator": Agent,
                "quant":        Agent,
                "strategist":   Agent,
                "futurist":     Agent,
                "cio":          Agent,
                "thai_summarizer": Agent,
            }
        """
        return {
            "interrogator": self.build_interrogator(),
            "quant":        self.build_quant(),
            "strategist":   self.build_business_strategist(),
            "futurist":     self.build_futurist(),
            "cio":          self.build_cio(),
            "thai_summarizer": self.build_thai_summarizer(),
        }