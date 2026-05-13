import streamlit as st
import pdfplumber
import docx
import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

st.set_page_config(page_title="ATS Resume Selector", page_icon="📄", layout="wide")
# ── Load Hugging Face Model ──────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()


# ── 1. TEXT EXTRACTION ──────────────────────────────────────────────────────

def extract_text_from_pdf(file) -> str:
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def extract_text_from_docx(file) -> str:
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return text.strip()


def extract_resume_text(file) -> str:
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif file.name.endswith(".docx"):
        return extract_text_from_docx(file)
    else:
        raise ValueError(f"Unsupported format: {file.name}")


# ── 2. TEXT PREPROCESSING ────────────────────────────────────────────────────

def preprocess_text(text: str) -> list:
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    clean_tokens = [
        word for word in tokens
        if word not in stop_words and word not in punctuation and word.isalpha()
    ]
    return clean_tokens


def extract_keywords(text: str) -> set:
    return set(preprocess_text(text))


# ── 3. SCORING ───────────────────────────────────────────────────────────────

def compute_keyword_score(resume_text: str, job_description: str) -> dict:
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(job_description)
    matched = resume_keywords.intersection(jd_keywords)
    missing = jd_keywords.difference(resume_keywords)
    score = round((len(matched) / len(jd_keywords)) * 100, 2) if jd_keywords else 0
    return {
        "keyword_score": score,
        "matched_keywords": sorted(matched),
        "missing_keywords": sorted(missing),
        "total_jd_keywords": len(jd_keywords),
        "total_matched": len(matched),
    }


def compute_semantic_score(resume_text: str, job_description: str) -> float:
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    jd_embedding = model.encode(job_description, convert_to_tensor=True)
    similarity = util.cos_sim(resume_embedding, jd_embedding)
    return round(float(similarity[0][0]) * 100, 2)


def compute_combined_score(resume_text: str, job_description: str) -> dict:
    keyword_result = compute_keyword_score(resume_text, job_description)
    semantic_score = compute_semantic_score(resume_text, job_description)
    combined_score = round(
        (keyword_result["keyword_score"] * 0.4) + (semantic_score * 0.6), 2
    )
    return {
        "combined_score": combined_score,
        "keyword_score": keyword_result["keyword_score"],
        "semantic_score": semantic_score,
        "matched_keywords": keyword_result["matched_keywords"],
        "missing_keywords": keyword_result["missing_keywords"],
        "total_jd_keywords": keyword_result["total_jd_keywords"],
        "total_matched": keyword_result["total_matched"],
    }


# ── 4. STREAMLIT UI ──────────────────────────────────────────────────────────

#st.set_page_config(page_title="ATS Resume Selector", page_icon="📄", layout="wide")

st.title("📄 ATS Resume Selector")
st.markdown("Upload resumes and paste a job description to rank candidates by ATS match score.")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload one or more resumes (PDF or DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

with col2:
    st.subheader("📋 Job Description")
    job_description = st.text_area(
        "Paste the job description here",
        height=250,
        placeholder="We are looking for a Machine Learning Engineer with Python, NLP, Deep Learning..."
    )

st.markdown("---")

if st.button("🚀 Analyse & Rank Resumes", use_container_width=True):

    if not uploaded_files:
        st.warning("⚠️ Please upload at least one resume.")
    elif not job_description.strip():
        st.warning("⚠️ Please paste a job description.")
    else:
        results = []
        progress = st.progress(0)
        status = st.empty()

        for i, file in enumerate(uploaded_files):
            status.text(f"🔍 Analysing {file.name}...")
            try:
                resume_text = extract_resume_text(file)
                result = compute_combined_score(resume_text, job_description)
                candidate_name = file.name.replace(".pdf", "").replace(".docx", "")

                results.append({
                    "Candidate":          candidate_name,
                    "Final Score (%)":    result["combined_score"],
                    "Keyword Score (%)":  result["keyword_score"],
                    "Semantic Score (%)": result["semantic_score"],
                    "Matched Keywords":   result["total_matched"],
                    "Total JD Keywords":  result["total_jd_keywords"],
                    "Missing Keywords":   ", ".join(result["missing_keywords"][:8]),
                })

            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {str(e)}")

            progress.progress((i + 1) / len(uploaded_files))

        status.text("✅ Analysis complete!")

        if results:
            df = pd.DataFrame(results).sort_values("Final Score (%)", ascending=False).reset_index(drop=True)
            df.index += 1
            df.index.name = "Rank"

            st.markdown("---")
            st.subheader("🏆 Ranked Results")
            st.dataframe(df, use_container_width=True)

            st.markdown("---")
            st.subheader(f"🥇 Top Candidate: {df.iloc[0]['Candidate']}")

            m1, m2, m3 = st.columns(3)
            m1.metric("Final Score",    f"{df.iloc[0]['Final Score (%)']}%")
            m2.metric("Keyword Score",  f"{df.iloc[0]['Keyword Score (%)']}%")
            m3.metric("Semantic Score", f"{df.iloc[0]['Semantic Score (%)']}%")

            st.markdown(f"**❌ Missing Keywords:** {df.iloc[0]['Missing Keywords']}")

            st.markdown("---")
            csv = df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="ats_rankings.csv",
                mime="text/csv",
                use_container_width=True
            )