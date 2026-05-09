import streamlit as st
import pandas as pd
from google import genai

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FairCheck AI – Bias Detector",
    page_icon="⚖️",
    layout="centered"
)

st.markdown("""
<style>
    .main { max-width: 800px; }
    .stAlert { border-radius: 10px; }
    h1 { color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

# ── Get API key from Streamlit secrets (hidden from users) ────────────────────
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    api_key = None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1>⚖️ FairCheck AI <span style='font-size:18px; color:gray;'>by Colon DoubleSlash</span></h1>", unsafe_allow_html=True)
st.subheader("Detect & Fix Bias in Your Dataset — Powered by Google Gemini")
st.markdown("---")

st.markdown("""
### 👋 Welcome!
This tool helps you find **hidden bias** in your data.

Bias means your data might be **unfair** to certain groups of people 
based on their gender, race, age, or other characteristics.

**How to use:**
1. Upload your CSV file (Excel-like data file)
2. Click **Analyse for Bias**
3. Get a full report in plain English — completely FREE!
""")

# ── File Upload ───────────────────────────────────────────────────────────────
st.markdown("### 📂 Upload your CSV file")
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Upload any CSV file containing people's data — hiring records, loan data, medical records, etc."
)

df = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df_display = df.head().astype(str)
        st.success(f"✅ File loaded successfully! — {df.shape[0]} rows, {df.shape[1]} columns")
        with st.expander("👀 Click to preview your data"):
            st.dataframe(df_display)
    except Exception as e:
        st.error(f"Could not read file: {e}")

# ── Analyse Button ────────────────────────────────────────────────────────────
st.markdown("### 🔍 Run Bias Analysis")
analyse_btn = st.button(
    "🚀 Analyse for Bias — FREE",
    use_container_width=True,
    type="primary"
)

if analyse_btn:
    if api_key is None:
        st.error("❌ App configuration error. Please contact the app owner.")
    elif df is None:
        st.error("❌ Please upload a CSV file first.")
    else:
        with st.spinner("🤖 Gemini AI is analysing your dataset for bias... please wait 20-30 seconds"):
            try:
                client = genai.Client(api_key=api_key)

                columns = list(df.columns)
                sample_data = df.head(10).to_csv(index=False)
                shape = df.shape

                cat_summary = ""
                for col in df.columns:
                    if df[col].dtype == "object" or df[col].nunique() <= 15:
                        counts = df[col].value_counts().head(5).to_dict()
                        cat_summary += f"\n- {col}: {counts}"

                prompt = f"""
You are an AI Fairness Expert. Analyse this dataset for bias.

DATASET: {shape[0]} rows, {shape[1]} columns
COLUMNS: {columns}

SAMPLE DATA:
{sample_data}

CATEGORY COUNTS:
{cat_summary}

Write a bias analysis report with these sections:

## 1. OVERALL BIAS RISK
Rate as LOW / MEDIUM / HIGH and explain in 2-3 sentences.

## 2. BIASED COLUMNS
For each suspicious column:
- Column name
- Type of bias (gender, race, age, proxy bias etc.)
- How it could hurt someone in real life

## 3. BIAS TYPES DETECTED
Which of these exist: Historical bias, Representation bias, Proxy bias, Sampling bias, Measurement bias

## 4. HOW TO FIX IT
Give 4-5 simple recommendations anyone can understand.

## 5. FAIRNESS SCORE
Score out of 10 and explanation.

Use plain simple English. No heavy technical words.
"""

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )

                result_text = response.text

                st.markdown("---")
                st.markdown("## 📋 Your Bias Analysis Report")
                st.markdown(result_text)
                st.markdown("---")

                st.download_button(
                    label="📥 Download Full Report",
                    data=result_text.encode("utf-8"),
                    file_name="bias_report.txt",
                    mime="text/plain"
                )

                st.success("✅ Analysis complete! Share this report with your team.")
                st.balloons()

            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<small>
⚖️ <b>FairCheck AI</b> — Making AI decisions fair for everyone.<br>
Powered by <b>Google Gemini AI</b> | Built for Google Solution Challenge 2026
</small>
""", unsafe_allow_html=True)
