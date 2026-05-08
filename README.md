# AI Stock Valuation Multi-Agent System

An AI-powered stock analysis pipeline using **CrewAI** multi-agent framework with **Google Gemini** LLM. Automatically generates in-depth investment reports and uploads them to **Google Drive** and **NotebookLM**.

## What It Does

Runs 6 AI agents sequentially to produce a full investment thesis:

| # | Agent | Output |
|---|-------|--------|
| 1 | **Interrogator** | 15–20 deep research questions |
| 2 | **Quant** | Financial metrics & valuation analysis |
| 3 | **Strategist** | Competitive positioning & moat analysis |
| 4 | **Futurist** | S-curve technology & future growth analysis |
| 5 | **CIO** | Final BUY / HOLD / PASS verdict with confidence level |
| 6 | **Thai Summarizer** | Full Thai-language summary of the CIO report |

After analysis completes, all 6 reports are automatically:
- Uploaded to **Google Drive** (`StockReports/<TICKER>/`)
- Added as sources to a **NotebookLM** notebook (`Stock Analysis — <TICKER>`)

---

## Requirements

- Python 3.12+
- [Google Gemini API key](https://aistudio.google.com/app/apikey)
- Google account (for Drive + NotebookLM)

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/StockMultiAgentProj.git
cd StockMultiAgentProj

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
playwright install chromium
```

---

## Configuration

### 1. Google Gemini API Key

Create a `.env` file in the project root:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get your key at: https://aistudio.google.com/app/apikey

### 2. Google Drive (auto-upload reports)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project → Enable **Google Drive API**
3. Go to **APIs & Services → Credentials** → Create **OAuth 2.0 Client ID** (Desktop app)
4. Download the JSON → rename it to `credentials.json` → place in project root
5. First run will open a browser for authentication → `token.json` saved automatically

### 3. NotebookLM (auto-add to notebook)

```bash
notebooklm login
```

A browser will open → sign in with your Google account → press Enter when the NotebookLM homepage loads. Auth is saved automatically for all future runs.

---

## Usage

```bash
source venv/bin/activate

python main.py --ticker TSLA
python main.py --ticker NVDA
python main.py --ticker AMD --company "Advanced Micro Devices"
```

### Output

Reports are saved to `./reports/`:
```
reports/
├── 01_interrogator_questions_TSLA.md
├── 02_quant_analysis_TSLA.md
├── 03_strategy_analysis_TSLA.md
├── 04_futurist_analysis_TSLA.md
├── 05_cio_investment_summary_TSLA.md
└── 06_thai_summary_TSLA.md
```

At the end of each run you'll see:
```
✅ Reports uploaded! Open in Drive:
   https://drive.google.com/drive/folders/...

✅ NotebookLM notebook ready:
   https://notebooklm.google.com/notebook/...
```

---

## Project Structure

```
StockMultiAgentProj/
├── main.py                  # Orchestration entry point (CLI)
├── agents.py                # All 6 CrewAI agent definitions
├── finance_tools.py         # Yahoo Finance tool (live data)
├── rag_engine.py            # RAG / vector store (ChromaDB)
├── drive_uploader.py        # Google Drive upload integration
├── notebooklm_uploader.py   # NotebookLM upload integration
├── requirements.txt
├── .env                     # (not committed) GOOGLE_API_KEY
├── credentials.json         # (not committed) Google OAuth credentials
└── reports/                 # (not committed) Generated reports
```

---

## Example Tickers

Any ticker supported by Yahoo Finance works:

```bash
python main.py --ticker AAPL   # Apple
python main.py --ticker NVDA   # NVIDIA
python main.py --ticker PLTR   # Palantir
python main.py --ticker IONQ   # IonQ
python main.py --ticker BTC-USD  # Bitcoin
```

---

## Tech Stack

- [CrewAI](https://github.com/joaomdmoura/crewAI) — multi-agent orchestration
- [Google Gemini](https://deepmind.google/technologies/gemini/) — LLM via LiteLLM
- [yfinance](https://github.com/ranaroussi/yfinance) — live stock data
- [ChromaDB](https://www.trychroma.com/) — vector store for RAG
- [notebooklm-py](https://github.com/teng-lin/notebooklm-py) — NotebookLM API
- [Google Drive API](https://developers.google.com/drive) — report storage
