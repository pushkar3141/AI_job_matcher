import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder

st.set_page_config(page_title="TalentVector AI Job Matcher", page_icon="🎯", layout="centered")
st.title("🎯 TalentVector AI Job Matcher")
st.markdown("Powered by Two-Stage Semantic Retrieval (Bi-Encoder + Cross-Encoder)")

@st.cache_resource(show_spinner="Loading AI Models... (This takes a minute on startup)")
def load_models():
    embedder = SentenceTransformer("intfloat/e5-base-v2")
    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return embedder, reranker_model

embedding_model, reranker = load_models()

def preprocess_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

@st.cache_resource(show_spinner="Building Search Index from Dataset...")
def prepare_system():
    df = pd.read_csv('cleaned_jobs_dataset.csv')

    for col in ["positionName", "company", "location", "jobType/0", "clean_description"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    df["combined_text"] = (
        "Job Title: " + df.get("positionName", "") + ". " +
        "Company: " + df.get("company", "") + ". " +
        "Location: " + df.get("location", "") + ". " +
        "Job Type: " + df.get("jobType/0", "") + ". " +
        "Description: " + df.get("clean_description", "")
    )
    df["combined_text"] = df["combined_text"].apply(preprocess_text)

    job_embeddings = embedding_model.encode(
        df["combined_text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    return df, job_embeddings

d, job_embeddings = prepare_system()

resume_input = st.text_area("Paste candidate resume or profile here...", height=150)

if st.button("Find Best Matches", type="primary"):
    if not resume_input.strip():
        st.warning("Please paste a resume first!")
    else:
        with st.spinner("Analyzing semantic context... 🤖"):
            query_clean = preprocess_text(resume_input)

            bge_query = "Represent this sentence for searching relevant jobs: " + query_clean
            query_embedding = embedding_model.encode(
                [bge_query],
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype("float32")

            scores = cosine_similarity(query_embedding, job_embeddings)[0]
            top_50_indices = np.argsort(scores)[::-1][:50].tolist()

            pairs = [[query_clean, d["combined_text"].iloc[idx]] for idx in top_50_indices]
            cross_scores = reranker.predict(pairs)

            scored_candidates = list(zip(top_50_indices, cross_scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            st.subheader("Top Recommendations:")
            for rank, (idx, score) in enumerate(scored_candidates[:5]):
                job = d.iloc[idx]

                apply_link = job.get('externalApplyLink', '#')
                if apply_link == 'unknown' or pd.isna(apply_link):
                    apply_link = job.get('url', '#')

                with st.container():
                    st.markdown(f"""
                    <div style="border-left: 4px solid #4F46E5; background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <h4 style="margin:0; color: #1f2937;">{str(job.get('positionName', 'Unknown Title')).title()}</h4>
                        <p style="margin: 5px 0 12px 0; color: #4b5563; font-size: 14px;">
                            🏢 {str(job.get('company', 'Unknown Company')).title()} | 
                            📍 {str(job.get('location', 'Unknown Location')).title()} | 
                            💼 {str(job.get('jobType/0', 'Unknown Type')).title()}
                        </p>
                        <span style="background-color: #D1FAE5; color: #065F46; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; margin-right: 15px;">
                            Relevance Score: {round(float(score) * 100, 2)}
                        </span>
                        <a href="{apply_link}" target="_blank" style="text-decoration: none; font-size: 12px; background-color: #4F46E5; color: white; padding: 5px 12px; border-radius: 5px; font-weight: bold;">Apply Here ↗</a>
                    </div>
                    """, unsafe_allow_html=True)
