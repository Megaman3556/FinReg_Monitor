import os
import json
from typing import Optional, List, Dict, Any

import pdfplumber
import pandas as pd
import streamlit as st
from google import genai  # Gemini API SDK
import re

# NLTK VADER
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is available
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except Exception:
    nltk.download("vader_lexicon")

# ==========================
#  CONFIG
# ==========================
DEFAULT_MODEL = "gemini-2.5-flash"

# Path to the uploaded sample in the conversation history (developer instruction)
SAMPLE_UPLOAD_PATH = "/mnt/data/INFOSYS_MDA.docx"

# ==========================
#  PDF / DOCX → TEXT EXTRACTOR
# ==========================
def extract_mda_text_from_pdf_filelike(file_like) -> Optional[str]:
    text_chunks = []
    with pdfplumber.open(file_like) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_chunks.append(page_text)
    if not text_chunks:
        return None
    return "\n".join(text_chunks).strip()

def extract_text_from_docx_path(path: str) -> Optional[str]:
    try:
        from docx import Document
    except Exception:
        raise RuntimeError("python-docx not installed. Install via: pip install python-docx")

    if not os.path.exists(path):
        return None
    doc = Document(path)
    paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n\n".join(paras).strip()

MDA_START_PATTERNS = [
    r"item\s+7\.*\s*management['’`s]* discussion and analysis",
    r"management['’`s]* discussion and analysis",
    r"management’s discussion and analysis",
    r"management and discussion of operations",
    r"management discussion and analysis",
]

MDA_END_PATTERNS = [
    r"item\s+7a\.*\s*.*?quantitative and qualitative",
    r"risk factors",
    r"financial statements",
    r"controls and procedures",
    r"critical accounting estimates",
    r"notes to the financial statements",
]

def find_mda_section(full_text: str) -> str:
    txt = re.sub(r"\s+", " ", full_text).strip()
    txt_low = txt.lower()
    start_idx, end_idx = None, None

    for pat in MDA_START_PATTERNS:
        m = re.search(pat, txt_low, flags=re.IGNORECASE)
        if m:
            start_idx = m.start()
            break

    for pat in MDA_END_PATTERNS:
        m = re.search(pat, txt_low, flags=re.IGNORECASE)
        if m:
            end_idx = m.start()
            break

    if start_idx is not None:
        if end_idx is None or end_idx <= start_idx:
            mda = txt[start_idx:]
        else:
            mda = txt[start_idx:end_idx]
        return re.sub(r"\s+", " ", mda).strip()

    fallback = re.search(r"\bmanagement\b", txt_low, flags=re.IGNORECASE)
    if fallback:
        idx = fallback.start()
        return re.sub(r"\s+", " ", txt[max(0, idx-1000): idx + 8000]).strip()

    return txt[:15000].strip()

# ==========================
#  GEMINI CLIENT
# ==========================
def get_gemini_client(api_key: Optional[str]) -> genai.Client:
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key missing. Set GEMINI_API_KEY or enter it in the UI.")
    return genai.Client(api_key=api_key)

def analyze_mda_with_llm(mda_text: str, api_key: Optional[str], model: str) -> Dict[str, Any]:
    client = get_gemini_client(api_key)
    system_msg = (
        "You are a senior financial risk analyst specializing in interpreting "
        "Management Discussion and Analysis sections of corporate filings."
    )
    user_prompt = f"""
You are given the Management Discussion and Analysis (MDA) text.

Your tasks:
1. Identify company-specific risks explicitly mentioned or implied in the MDA.
2. Assign each risk to one of:
- Market & Economic Risk
- Operational & Physical Risk
- Technology Risk
- Crime & Security Risk
- Natural Hazard & Event Risk
- Other Strategic / External Risks

Return ONLY valid JSON:
{{
  "risks": [
    {{
      "title": "...",
      "risk_area": "...",
      "time_horizon": "short/medium/long"
    }}
  ]
}}
MDA TEXT:
\"\"\"{mda_text}\"\"\""""

    response = client.models.generate_content(
        model=model,
        contents=[{"role":"user", "parts":[{"text":system_msg + "\n\n" + user_prompt}]}],
    )
    raw = response.text
    try:
        return json.loads(raw)
    except Exception:
        s = raw.find("{"); e = raw.rfind("}")
        if s != -1 and e != -1:
            return json.loads(raw[s:e+1])
        raise ValueError("Gemini returned invalid JSON:\n" + raw)

# ==========================
#  Sentiment helpers
# ==========================
vader = SentimentIntensityAnalyzer()

def compute_compound_for_risk(item: Dict[str, Any]) -> float:
    text_parts = []
    if item.get("title"):
        text_parts.append(item["title"])
    ev = item.get("evidence", [])
    if isinstance(ev, list) and ev:
        text_parts.append(ev[0])
    combined = " . ".join(text_parts)
    if not combined:
        return 0.0
    return float(vader.polarity_scores(combined)["compound"])

def build_risks_dataframe_with_sentiment(risks: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(risks)
    df["compound"] = df.apply(lambda r: compute_compound_for_risk(r.to_dict()), axis=1)
    return df

# ==========================
#  TABLE STYLING
# ==========================
def style_and_center_dataframe(df: pd.DataFrame, top_n: int = 5) -> str:
    if df.empty:
        return "<div>No risks to display</div>"

    df_sorted = df.sort_values("compound").reset_index(drop=True)
    worst_idx = df_sorted.head(top_n).index.tolist()

    styler = df_sorted.style.set_properties(**{
        'text-align':'center',
        'font-family':'Arial'
    }).set_table_styles([{'selector':'th','props':[('text-align','center')]}])

    def highlight(row):
        return ['background-color:#870404' if row.name in worst_idx else '' for _ in row]

    styler = styler.apply(highlight, axis=1).format({"compound":"{:.3f}"})

    html = styler.to_html()
    return f"""
    <div style="display:flex; justify-content:center;">
        <div style="width:80%;">{html}</div>
    </div>
    """

# ==========================
#  STREAMLIT UI
# ==========================
def main():
    st.set_page_config(page_title="FinReg Monitor", layout="wide")
    st.title("📘 FinReg Monitor")

    st.write("Upload an **MDA file** for automated company-specific risk extraction and sentiment ranking.")

    st.sidebar.header("FinReg Analysis Settings")
    st.sidebar.write("Analyze using Gemini to identify company-specific risks and tabulate them.")
    st.sidebar.write("Compute VADER sentiment and display **Top 5 most negative risks**.")
    
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
    model_name = st.sidebar.text_input("Gemini Model", value=DEFAULT_MODEL)

    uploaded_pdf = st.sidebar.file_uploader("Upload MDA PDF", type=["pdf"])
    use_sample = st.sidebar.button("Use Sample MDA (from session upload)")

    mda_text = None

    if use_sample:
        if os.path.exists(SAMPLE_UPLOAD_PATH):
            full = extract_text_from_docx_path(SAMPLE_UPLOAD_PATH)
            mda_text = find_mda_section(full)
            st.success("Loaded sample MDA file.")
        else:
            st.error(f"Sample not found at: {SAMPLE_UPLOAD_PATH}")

    if uploaded_pdf is not None and mda_text is None:
        import io
        raw = uploaded_pdf.read()
        mda_raw = extract_mda_text_from_pdf_filelike(io.BytesIO(raw))
        if mda_raw:
            mda_text = find_mda_section(mda_raw)
            st.success("Uploaded MDA processed.")
        else:
            st.error("Failed to extract text from uploaded MDA.")

    if not mda_text:
        st.info("Upload an MDA file or use the sample.")
        return

    st.subheader("Extracted MDA (Preview)")
    st.code(mda_text[:3000] + ("...\n(TRUNCATED)" if len(mda_text) > 3000 else ""), language="text")

    if st.button("Run FinReg Analysis"):
        try:
            result = analyze_mda_with_llm(mda_text, api_key, model_name)
        except Exception as e:
            st.error(f"Gemini error: {e}")
            return

        risks = result.get("risks", [])
        if not risks:
            st.warning("No risks detected.")
            return

        df_risks = build_risks_dataframe_with_sentiment(risks)
        df_sorted = df_risks.sort_values("compound").reset_index(drop=True)

        html = style_and_center_dataframe(df_sorted, top_n=5)
        st.markdown(html, unsafe_allow_html=True)

        st.subheader("Top 5 Most Negative Risks")
        st.table(df_sorted.head(5)[["title","risk_area","time_horizon","compound"]])

    st.caption("Powered by Gemini + VADER sentiment analysis.")

if __name__ == "__main__":
    main()
