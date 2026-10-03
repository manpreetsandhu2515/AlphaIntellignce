import os
import csv
import io
import sqlite3
import requests
import streamlit as st
from datetime import datetime
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="ALPHAFUND TERMINAL",
    page_icon="▸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
section.main > div {
    background-color: #080808 !important;
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    color: #ff9900 !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background-color: #040404 !important;
    border-right: 1px solid #3a1f00 !important;
}
[data-testid="stSidebar"] * {
    color: #ff9900 !important;
    font-family: 'Share Tech Mono', monospace !important;
}
[data-testid="stSidebar"] input {
    background: #0c0c0c !important;
    border: 1px solid #3a1f00 !important;
    color: #ff9900 !important;
    border-radius: 0 !important;
    font-family: 'Share Tech Mono', monospace !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #0c0c0c !important;
    border: 1px solid #3a1f00 !important;
    border-radius: 0 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: #0c0c0c !important;
    border: 1px solid #ff6600 !important;
    color: #ff9900 !important;
    border-radius: 0 !important;
    width: 100%;
    letter-spacing: 2px;
    font-family: 'Share Tech Mono', monospace !important;
    transition: all 0.15s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #ff6600 !important;
    color: #000 !important;
}
[data-testid="stSidebar"] hr { border-color: #2a1200 !important; }

/* ── METRICS ── */
[data-testid="stMetric"] {
    background: #0c0c0c !important;
    border: 1px solid #3a1f00 !important;
    padding: 14px 18px !important;
    border-radius: 0 !important;
}
[data-testid="stMetricLabel"] p {
    color: #7a4000 !important;
    font-size: 9px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    font-family: 'Share Tech Mono', monospace !important;
}
[data-testid="stMetricValue"] {
    color: #ff9900 !important;
    font-size: 16px !important;
    font-family: 'Share Tech Mono', monospace !important;
}
[data-testid="stMetricDelta"] { color: #00bb44 !important; font-size: 10px !important; }

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: #0a0a0a !important;
    border: 1px solid #2a1200 !important;
    border-radius: 0 !important;
}
details summary {
    color: #7a4000 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
}
details summary:hover { color: #ff9900 !important; }

/* ── CODE ── */
pre, code {
    background: #050505 !important;
    color: #00bb44 !important;
    font-family: 'Share Tech Mono', monospace !important;
    border: 1px solid #1a0a00 !important;
    border-radius: 0 !important;
}

/* ── STATUS ── */
[data-testid="stStatus"] {
    background: #0a0a0a !important;
    border: 1px solid #ff6600 !important;
    border-radius: 0 !important;
}
[data-testid="stStatus"] * {
    color: #ff9900 !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputTextArea"] textarea {
    background: #0c0c0c !important;
    color: #ff9900 !important;
    font-family: 'Share Tech Mono', monospace !important;
    border: none !important;
    caret-color: #ff9900 !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #3a1f00 !important; }
[data-testid="stBottom"],
[data-testid="stBottom"] > div {
    background: #080808 !important;
    border-top: 1px solid #2a1200 !important;
}
div[class*="stChatInput"] {
    background: #0c0c0c !important;
    border: 1px solid #ff6600 !important;
    border-radius: 0 !important;
}

/* ── MARKDOWN ── */
[data-testid="stMarkdownContainer"] p { color: #e0a060 !important; }
[data-testid="stMarkdownContainer"] strong { color: #ff9900 !important; }
[data-testid="stMarkdownContainer"] li { color: #e0a060 !important; }

/* ── ALERT ── */
[data-testid="stAlert"] {
    background: #120500 !important;
    border: 1px solid #ff6600 !important;
    border-radius: 0 !important;
    color: #ff9900 !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; background: #050505; }
::-webkit-scrollbar-thumb { background: #3a1f00; }

/* ── DROPDOWN ── */
[data-baseweb="popover"] ul {
    background: #0c0c0c !important;
    border: 1px solid #3a1f00 !important;
}
[data-baseweb="popover"] li { color: #ff9900 !important; background: #0c0c0c !important; }
[data-baseweb="popover"] li:hover { background: #1a0800 !important; }

hr { border-color: #1a0a00 !important; }

/* ── HIDE STREAMLIT BRANDING ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── CUSTOM CHAT CARDS ── */
.af-user-card {
    background: #0d0800;
    border: 1px solid #3a1f00;
    border-left: 3px solid #ff6600;
    padding: 14px 18px;
    margin: 10px 0;
    font-family: 'Share Tech Mono', monospace;
}
.af-user-card .af-label {
    font-size: 9px;
    letter-spacing: 3px;
    color: #ff6600;
    margin-bottom: 8px;
    display: block;
}
.af-user-card .af-text {
    color: #ffcc88;
    font-size: 14px;
    line-height: 1.6;
}

.af-ai-card {
    background: #050f05;
    border: 1px solid #0a2a0a;
    border-left: 3px solid #00bb44;
    padding: 14px 18px;
    margin: 10px 0;
    font-family: 'Share Tech Mono', monospace;
}
.af-ai-card .af-label {
    font-size: 9px;
    letter-spacing: 3px;
    color: #00bb44;
    margin-bottom: 8px;
    display: block;
}
.af-ai-card .af-text {
    color: #aaddaa;
    font-size: 14px;
    line-height: 1.7;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────
now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
st.markdown(f"""
<div style="
    display:flex; justify-content:space-between; align-items:center;
    background:#040404; border:1px solid #3a1f00;
    border-left:4px solid #ff6600;
    padding:18px 28px; margin-bottom:24px;">
  <div>
    <div style="font-family:'Share Tech Mono',monospace; font-size:22px;
                color:#ff9900; letter-spacing:8px;">
      ▸ ALPHAFUND TERMINAL
    </div>
    <div style="font-family:'Share Tech Mono',monospace; font-size:10px;
                color:#7a4000; letter-spacing:3px; margin-top:4px;">
      SQL ENGINE  ·  VECTOR SEARCH  ·  AI ORCHESTRATION
    </div>
  </div>
  <div style="text-align:right;">
    <div style="font-family:'Share Tech Mono',monospace; font-size:13px; color:#ff6600;">
      {now}
    </div>
    <div style="font-family:'Share Tech Mono',monospace; font-size:9px;
                color:#3a1f00; letter-spacing:2px; margin-top:4px;">
      ● LIVE
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:11px;
                color:#7a4000;letter-spacing:3px;padding:4px 0 12px;">
    ▸ SYSTEM CONFIG
    </div>""", unsafe_allow_html=True)

    api_key = st.text_input(
        "OPENROUTER KEY",
        type="password",
        value=os.environ.get("OPENROUTER_API_KEY", ""),
        help="openrouter.ai/keys",
    )
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key

    model_choice = st.selectbox(
        "ROUTING MODEL",
        options=[
            "google/gemini-2.0-flash-001",
            "anthropic/claude-3.5-haiku",
            "openai/gpt-4o-mini",
            "mistralai/mistral-nemo",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:10px;
                color:#7a4000;letter-spacing:1px;line-height:2.0;">
    <span style="color:#ff6600;">[ SQL ]</span>  S&P 500 · GITHUB CSV<br>
    <span style="color:#ff6600;">[ VEC ]</span>  Q4 TRANSCRIPTS · FAISS<br>
    <br>
    <span style="color:#3a1f00;">COVERED TICKERS</span><br>
    MSFT · AAPL · GOOGL<br>
    NVDA · AMZN
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("[ CLEAR SESSION ]"):
        st.session_state.messages = []
        st.rerun()

# ─── TRANSCRIPTS ─────────────────────────────────────────────────────────────
TRANSCRIPTS = {
    "MSFT_Q4_2024.md": """Microsoft Corporation — Q4 FY2024 Earnings Call
CEO Satya Nadella: We are accelerating our AI infrastructure build-out, allocating an additional $2 billion
to Azure GPU clusters this quarter alone. Copilot integrations across Microsoft 365 have driven a 40% increase
in enterprise subscriptions. Azure revenue grew 29% year-over-year, with AI services contributing 8 percentage
points of that growth. We have made AI the core of every product layer.
CFO Amy Hood: Capital expenditures for the quarter were $14 billion. Operating income increased 15% to
$27.9 billion. Commercial cloud revenue reached $36.8 billion, up 22% year-over-year. Net income was $22.1B.""",

    "AAPL_Q4_2024.md": """Apple Inc. — Q4 FY2024 Earnings Call
CEO Tim Cook: iPhone 16 Pro demand has significantly exceeded our expectations. Apple Intelligence features
are driving the strongest upgrade cycle we have seen in three years. Services achieved an all-time record
of $25 billion this quarter.
CFO Luca Maestri: Revenue was $94.9 billion, up 6% year-over-year. Gross margin was 46.2%.
We returned $29 billion to shareholders through dividends and share repurchases.""",

    "GOOGL_Q4_2024.md": """Alphabet Inc. — Q4 2024 Earnings Call
CEO Sundar Pichai: Search with AI Overviews is now serving over 1 billion queries per day. Google Cloud
reached $12 billion in quarterly revenue — 35% year-over-year growth. Gemini Ultra is powering enterprise AI.
CFO Anat Ashkenazi: Total revenue was $96.5 billion, up 15%. Operating income was $28.5 billion.
YouTube advertising revenue grew 14% to $10.5 billion. CapEx was $13.7 billion.""",

    "NVDA_Q4_2024.md": """NVIDIA Corporation — Q4 FY2025 Earnings Call
CEO Jensen Huang: Data center revenue reached $35.6 billion — up 409% year-over-year. Demand for Blackwell
GPUs has far exceeded our ability to supply. Physical AI and robotics represent a multi-trillion-dollar
opportunity. We shipped over 1 million Blackwell units this quarter.
CFO Colette Kress: Total revenue was $39.3 billion, up 265% year-over-year. GAAP gross margin was 74.6%.
Operating cash flow was $27.7 billion.""",

    "AMZN_Q4_2024.md": """Amazon.com Inc. — Q4 2024 Earnings Call
CEO Andy Jassy: AWS revenue grew 19% year-over-year to $28.8 billion. Customers are using Amazon Bedrock
to build production-grade generative AI applications at scale. Advertising generated $17.3 billion, up 18%.
CFO Brian Olsavsky: Net sales were $187.8 billion, up 10%. Operating income was $21.2 billion.
Capital expenditures in 2024 totaled $83 billion for AWS and AI infrastructure.""",
}

# ─── SQL DATABASE ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="LOADING S&P 500 FROM GITHUB...")
def setup_sql_database():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE financials (
        ticker TEXT PRIMARY KEY, name TEXT, sector TEXT,
        market_cap REAL, ebitda REAL, price REAL, pe_ratio REAL, eps REAL)""")
    data_source = "FALLBACK"
    try:
        resp = requests.get(
            "https://raw.githubusercontent.com/datasets/"
            "s-and-p-500-companies-financials/main/data/constituents-financials.csv",
            timeout=15)
        resp.raise_for_status()
        rows = []
        for row in csv.DictReader(io.StringIO(resp.text)):
            try:
                rows.append((
                    row["Symbol"].strip(), row["Name"].strip(), row["Sector"].strip(),
                    float(row.get("Market Cap") or 0), float(row.get("EBITDA") or 0),
                    float(row.get("Price") or 0), float(row.get("Price/Earnings") or 0),
                    float(row.get("Earnings/Share") or 0),
                ))
            except (ValueError, KeyError):
                continue
        cursor.executemany("INSERT OR REPLACE INTO financials VALUES (?,?,?,?,?,?,?,?)", rows)
        data_source = f"LIVE · {len(rows)} COS"
    except Exception:
        cursor.executemany("INSERT OR REPLACE INTO financials VALUES (?,?,?,?,?,?,?,?)", [
            ("MSFT","Microsoft Corporation","Information Technology",3.02e12,1.32e11,415.0,36.2,11.50),
            ("AAPL","Apple Inc.","Information Technology",3.45e12,1.29e11,220.0,33.4,6.43),
            ("GOOGL","Alphabet Inc.","Communication Services",2.10e12,1.05e11,178.0,24.1,7.40),
            ("NVDA","NVIDIA Corporation","Information Technology",3.30e12,6.50e10,875.0,68.5,12.77),
            ("AMZN","Amazon.com Inc.","Consumer Discretionary",2.15e12,8.50e10,205.0,48.3,4.30),
            ("META","Meta Platforms Inc.","Communication Services",1.52e12,5.80e10,598.0,27.2,22.10),
            ("TSLA","Tesla Inc.","Consumer Discretionary",8.00e11,1.17e10,250.0,78.6,3.22),
            ("JPM","JPMorgan Chase & Co.","Financials",5.80e11,6.50e10,198.0,12.4,15.96),
        ])
        data_source = "FALLBACK · 8 COS"
    conn.commit()
    return conn, data_source


@st.cache_resource(show_spinner="BUILDING FAISS INDEX...")
def setup_faiss_vectorstore(_key: str):
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    docs = []
    for fname, content in TRANSCRIPTS.items():
        ticker = fname.split("_")[0]
        docs.extend(splitter.create_documents([content], metadatas=[{"ticker": ticker}]))
    return FAISS.from_documents(docs, embedder)


# ─── TOOLS ────────────────────────────────────────────────────────────────────
def build_tools(conn, vectorstore):
    @tool
    def query_sp500_financials(sql_query: str) -> str:
        """
        Executes a SQL query against the S&P 500 financials database.
        Schema: financials(ticker TEXT, name TEXT, sector TEXT, market_cap REAL,
                           ebitda REAL, price REAL, pe_ratio REAL, eps REAL)
        market_cap is raw USD. Tickers are uppercase e.g. 'MSFT'.
        """
        try:
            cursor = conn.cursor()
            cursor.execute(sql_query)
            col_names = [d[0] for d in cursor.description]
            rows = [dict(zip(col_names, r)) for r in cursor.fetchall()]
            return str(rows) if rows else "No rows returned."
        except Exception as e:
            return f"SQL Error: {e}"

    @tool
    def search_earnings_transcripts(query: str) -> str:
        """
        Semantic search across Q4 2024 earnings call transcripts for
        MSFT, AAPL, GOOGL, NVDA, AMZN.
        Use for CEO/CFO quotes, AI strategy, CapEx, revenue guidance.
        """
        docs = vectorstore.similarity_search(query, k=3)
        return "\n\n---\n\n".join(
            f"[{d.metadata.get('ticker')}] {d.page_content.strip()}" for d in docs
        )

    return [query_sp500_financials, search_earnings_transcripts]


# ─── AGENT ────────────────────────────────────────────────────────────────────


def run_agent(user_prompt: str, tools: list, model: str) -> tuple[str, list[dict]]:
    """Create the LLM and invoke it with tools, returning (answer, tool_calls).
    If the selected OpenRouter model has no endpoints, retry once with a fallback model.
    """
    fallback_model = 'openai/gpt-4o-mini'

    def _invoke_with_model(chosen_model):
        try:
            llm = ChatOpenAI(
                model=chosen_model, temperature=0.0,
                openai_api_key=os.environ["OPENROUTER_API_KEY"],
                openai_api_base="https://openrouter.ai/api/v1",
                default_headers={"HTTP-Referer": "https://alphafund.app", "X-Title": "AlphaFund"},
            )
            llm_with_tools = llm.bind_tools(tools)
        except Exception as e:
            return None, None, f"LLM initialization error: {e}"

        tool_map = {t.name: t for t in tools}
        messages = [HumanMessage(content=user_prompt)]
        tool_calls_log = []

        try:
            # This is agentic reasoning -- the LLM decides what tools to call, observes results, and continues until it has enough info 
            while True:
                ai_msg = llm_with_tools.invoke(messages) # LLM responsds
                messages.append(ai_msg)
                if not ai_msg.tool_calls:
                    return ai_msg.content, tool_calls_log, None # No more tools -> return answer
                for tc in ai_msg.tool_calls:
                    tool_calls_log.append({"name": tc["name"], "args": tc["args"]}) #Execute tool
                    output = tool_map[tc["name"]].invoke(tc["args"]) if tc["name"] in tool_map else ""
                    messages.append(ToolMessage(tool_call_id=tc["id"], content=str(output))) #Feed result back
        except Exception as e:
            return None, tool_calls_log, f"LLM invocation error: {e}"

    # First attempt with requested model
    answer, tool_calls, err = _invoke_with_model(model)
    if err and ('No endpoints found' in err or '404' in err) and model != fallback_model:
        # Retry once with fallback
        answer, tool_calls, err2 = _invoke_with_model(fallback_model)
        if err2:
            # both failed — return combined error
            return f"Primary model error: {err} | Fallback model error: {err2}", tool_calls or []
        return answer, tool_calls or []
    if err:
        return err, tool_calls or []
    return answer, tool_calls or []

# ─── GATE ─────────────────────────────────────────────────────────────────────
if not os.environ.get("OPENROUTER_API_KEY"):
    st.markdown("""
    <div style="background:#0a0400;border:1px solid #ff6600;border-left:4px solid #ff6600;
                padding:24px 28px;font-family:'Share Tech Mono',monospace;">
      <div style="color:#ff6600;font-size:11px;letter-spacing:3px;margin-bottom:10px;">
        ⚠  AUTHENTICATION REQUIRED
      </div>
      <div style="color:#e0a060;font-size:13px;line-height:1.8;">
        Enter your OpenRouter API key in the sidebar to initialize the terminal.<br>
        <span style="color:#3a1f00;">→ openrouter.ai/keys</span>
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ─── LOAD ─────────────────────────────────────────────────────────────────────
conn, data_source = setup_sql_database()
vectorstore = setup_faiss_vectorstore(os.environ["OPENROUTER_API_KEY"])
tools = build_tools(conn, vectorstore)

# ─── STATUS STRIP ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("SQL DATABASE", "S&P 500", data_source)
c2.metric("VECTOR STORE", "FAISS", f"{len(TRANSCRIPTS)} TRANSCRIPTS")
c3.metric("EMBEDDINGS", "MiniLM-L6", "LOCAL · FREE")
c4.metric("LLM ROUTER", model_choice.split("/")[-1].upper()[:16], "OPENROUTER")

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ─── SAMPLE QUERIES ───────────────────────────────────────────────────────────
with st.expander("▸  SAMPLE QUERIES"):
    st.code("""[ BOSS FIGHT · DUAL TOOL ]
What is the exact market cap of Microsoft (MSFT), and what did their
CEO say about AI infrastructure investments in the latest earnings call?

[ MULTI-STOCK ]
Compare NVDA and AAPL market caps, then find what each CEO said about revenue.

[ PURE SQL ]
Which sector has the highest average EBITDA? Show the top 3 sectors.

[ PURE TRANSCRIPT ]
What did Amazon's CFO say about capital expenditure in 2024?

[ CROSS-REFERENCE ]
What is NVDA's P/E ratio, and does their earnings call justify the valuation?""")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ─── CHAT STATE ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── RENDER HISTORY ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="af-user-card">
          <span class="af-label">▸ ANALYST QUERY</span>
          <div class="af-text">{msg["content"]}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="af-ai-card">
          <span class="af-label">▸ ALPHA-AI RESPONSE</span>
          <div class="af-text">{msg["content"]}</div>
        </div>""", unsafe_allow_html=True)
        if msg.get("tool_calls"):
            with st.expander("▸  ROUTING LOG"):
                for tc in msg["tool_calls"]:
                    st.code(f"TOOL : {tc['name']}\nARGS : {tc['args']}", language="yaml")

# ─── INPUT ────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("ENTER QUERY  ·  e.g. MSFT market cap + CEO AI strategy..."):

    # Show user card immediately
    st.markdown(f"""
    <div class="af-user-card">
      <span class="af-label">▸ ANALYST QUERY</span>
      <div class="af-text">{prompt}</div>
    </div>""", unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Run agent + show result
    status = st.status("▸ ROUTING QUERY TO DATABASES...", expanded=True)
    try:
        with status:
            st.write(f"MODEL  : {model_choice}")
            answer, tool_calls = run_agent(prompt, tools, model_choice)
            if tool_calls:
                for tc in tool_calls:
                    st.write(f"FIRED  : {tc['name']}")
        status.update(label="▸ QUERY COMPLETE", state="complete", expanded=False)

        st.markdown(f"""
        <div class="af-ai-card">
          <span class="af-label">▸ ALPHA-AI RESPONSE</span>
          <div class="af-text">{answer}</div>
        </div>""", unsafe_allow_html=True)

        if tool_calls:
            with st.expander("▸  ROUTING LOG"):
                for tc in tool_calls:
                    st.code(f"TOOL : {tc['name']}\nARGS : {tc['args']}", language="yaml")

    except Exception as e:
        status.update(label="▸ ERROR", state="error", expanded=False)
        answer = f"SYSTEM ERROR: {e}"
        tool_calls = []
        st.markdown(f"""
        <div style="background:#0a0000;border:1px solid #cc0000;border-left:3px solid #cc0000;
                    padding:14px 18px;font-family:'Share Tech Mono',monospace;color:#ff4444;">
          ⚠ {answer}
        </div>""", unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "tool_calls": tool_calls
    })
