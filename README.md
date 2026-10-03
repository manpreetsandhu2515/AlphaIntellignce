# Alpha Intelligence
### Month 1 Capstone — AI Engineering Bootcamp

A unified intelligence dashboard for quantitative analysts. Type one question and the AI simultaneously queries a **structured SQL database** (S&P 500 financials) and an **unstructured vector store** (Q4 earnings call transcripts), then fuses both results into a single executive summary.

---

## What This App Does

| Layer | Technology | Data Source |
|---|---|---|
| SQL Engine | SQLite (in-memory) | S&P 500 CSV — pulled live from GitHub |
| Vector Search | FAISS + MiniLM embeddings | Q4 2024 earnings transcripts (MSFT, AAPL, GOOGL, NVDA, AMZN) |
| AI Orchestration | LangChain `bind_tools` agentic loop | OpenRouter API |
| Frontend | Streamlit | — |

---

## PART 1 — Run It Locally

### Step 1 — Get Your OpenRouter API Key

1. Go to **[openrouter.ai/keys](https://openrouter.ai/keys)**
2. Sign up for a free account (no billing required)
3. Click **Create Key**
4. Copy the key — it looks like `sk-or-v1-xxxxxxxx...`

---

### Step 2 — Clone the Repo

Open your terminal and run:

```bash
git clone https://github.com/YOUR-USERNAME/AlphaIntelligence.git
cd alphafund-terminal
```

> Replace `YOUR-USERNAME` with your actual GitHub username.

---

### Step 3 — Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Mac / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

---

### Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs everything — Streamlit, LangChain, FAISS, embeddings. Takes about 2 minutes on first run.

---

### Step 5 — Set Your API Key

**Option A — Paste it in the sidebar (easiest for beginners)**

Skip this step. Just run the app and paste your key into the sidebar input field when prompted.

**Option B — Environment variable (recommended)**

```bash
# Mac / Linux
cp .env.example .env
```

```bash
# Windows
copy .env.example .env
```

Open the `.env` file in any text editor and replace the placeholder:

```
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
```

Save the file.

> **IMPORTANT:** Never share or commit your `.env` file. It is already blocked by `.gitignore`.

---

### Step 6 — Run the App

```bash
streamlit run app.py
```

Your browser will open automatically at **http://localhost:8501**

On first launch the app will:
1. Pull the live S&P 500 CSV from GitHub → load into SQLite
2. Build the FAISS vector index from earnings transcripts
3. Show the terminal UI — ready for queries

---

### Step 7 — Choose Your Model

In the **sidebar**, use the **ROUTING MODEL** dropdown:

| Model | Speed | Best For |
|---|---|---|
| `google/gemini-2.0-flash-001` | Fast | **Start here — default** |
| `anthropic/claude-3.5-haiku` | Medium | Best answer quality |
| `openai/gpt-4o-mini` | Fast | Reliable fallback |
| `mistralai/mistral-nemo` | Fastest | Simple SQL-only queries |

---

### Step 8 — Test with the Boss-Fight Query

Paste this into the chat input:

```
What is the exact market cap of Microsoft (MSFT), and what did their CEO
say about AI infrastructure investments in the latest earnings call?
```

**What should happen:**
- Agent fires `query_sp500_financials` → gets MSFT market cap from SQL
- Agent fires `search_earnings_transcripts` → finds Satya Nadella's quote from FAISS
- Both results fused into one polished answer

Click **▸ ROUTING LOG** below the response to see which tools fired.

---

## PART 2 — Deploy to Streamlit Cloud (Share with the World)

### Step 1 — Create a GitHub Repo

1. Go to **[github.com/new](https://github.com/new)**
2. Name it `AlphaIntelligence`
3. Set it to **Public**
4. Do **NOT** tick "Add README" — we already have one
5. Click **Create repository**

---

### Step 2 — Push Your Code

Run these commands inside your project folder:

```bash
# Initialise git
git init

# Stage all files
git add app.py requirements.txt README.md .env.example .gitignore

# DO NOT run: git add .env  ← never add this

# Commit
git commit -m "Initial commit: AlphaFund Terminal"

# Connect to GitHub (replace YOUR-USERNAME)
git remote add origin https://github.com/YOUR-USERNAME/AlphaIntelligence.git

# Push
git branch -M main
git push -u origin main
```

---

### Step 3 — Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with your GitHub account
3. Click **New app**
4. Fill in:
   - **Repository:** `YOUR-USERNAME/AlphaIntelligence`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Advanced settings**
6. Under **Secrets**, paste this (replace with your real key):

```toml
OPENROUTER_API_KEY = "sk-or-v1-your-key-here"
```

7. Click **Deploy**

Your app will be live at:
```
https://YOUR-USERNAME-alphafund-terminal-app-xxxx.streamlit.app
```

---

## Project Structure

```
alphafund-terminal/
│
├── app.py               ← Main Streamlit app (all logic lives here)
├── requirements.txt     ← Python dependencies
├── .env.example         ← API key template (safe to commit)
├── .gitignore           ← Keeps your .env and cache out of git
└── README.md            ← This file
```

---

## How the AI Routing Works

```
User types a question
        │
        ▼
LLM receives question + 2 tools bound via bind_tools()
        │
        ├── Needs structured data?  → fires query_sp500_financials(sql)
        ├── Needs qualitative data? → fires search_earnings_transcripts(query)
        │         (both can fire in the same round)
        ▼
Tool results added to message history
        │
        ▼
LLM reads all results → writes final fused answer
        │
        ▼
Rendered in Streamlit terminal UI
```

This is a **two-pass agentic loop** — not a rigid if/else router. The AI decides which tools to use based on the question. If both are needed, both fire simultaneously.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `401 Unauthorized` | API key is wrong or expired — generate a new one at openrouter.ai/keys |
| `429 Rate Limited` | Hit free tier limit — wait 1 minute or switch to a different model |
| `400 Invalid model ID` | Model name typo — copy exact model string from the dropdown |
| `SQL Error: no such table` | Database failed to load — check internet connection and re-run |
| `No rows returned` | Ticker not in dataset — try MSFT, AAPL, GOOGL, NVDA, AMZN |
| App shows fallback data | GitHub CSV unreachable — app loaded 8 hardcoded stocks instead |
| White screen on deploy | Check Streamlit Cloud logs — most likely the secret key is missing |

---

## Sample Queries to Try

```
# Dual tool — the boss fight
What is the exact market cap of Microsoft (MSFT), and what did their CEO
say about AI infrastructure investments in the latest earnings call?

# Multi-stock comparison
Compare NVDA and AAPL market caps, then find what each CEO said about revenue growth.

# Pure SQL
Which sector has the highest average EBITDA? Show the top 3 sectors.

# Pure transcript
What did Amazon's CFO say about capital expenditure in 2024?

# Cross-reference valuation
What is NVDA's P/E ratio, and does their earnings call justify the valuation?
```

---

*Built for the AI Engineering Bootcamp — Month 1 Capstone*
