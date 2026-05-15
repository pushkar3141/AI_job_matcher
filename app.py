import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer, CrossEncoder

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="TalentVector AI Job Matcher", page_icon="🎯", layout="centered")

st.title("🎯 TalentVector AI Job Matcher")
st.markdown("Powered by Two-Stage Semantic Retrieval (Bi-Encoder + Cross-Encoder)")

# ==========================================
# 1. CACHED MODEL LOADING
# ==========================================
@st.cache_resource(show_spinner="Loading AI Models... (This takes a minute on startup)")
def load_models():
    embedder = SentenceTransformer("intfloat/e5-base-v2")
    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return embedder, reranker_model

embedding_model, reranker = load_models()

# ==========================================
# 2. CACHED DATA & EMBEDDINGS
# ==========================================
def preprocess_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

@st.cache_resource(show_spinner="Building Vector Search Index from Cleaned Data...")
def prepare_system():
    # Load your highly-optimized cleaned CSV
    df = pd.read_csv('cleaned_jobs_dataset.csv')
    
    # Fill any missing values securely
    for col in ["positionName", "company", "location", "jobType/0", "clean_description"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # Combine the pre-cleaned text with the metadata
    df["combined_text"] = (
        "Job Title: " + df.get("positionName", "") + ". " +
        "Company: " + df.get("company", "") + ". " +
        "Location: " + df.get("location", "") + ". " +
        "Job Type: " + df.get("jobType/0", "") + ". " +
        "Description: " + df.get("clean_description", "")
    )

    # Pre-compute all embeddings for Stage 1 FAISS retrieval
    job_embeddings = embedding_model.encode(
        df["combined_text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # Build High-Speed FAISS Index
    dim = job_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(job_embeddings)
    
    return df, index

d, faiss_index = prepare_system()

# ==========================================
# 3. USER INTERFACE & SEARCH LOGIC
# ==========================================
resume_input = st.text_area("Paste candidate resume or profile here...", height=150)

if st.button("Find Best Matches", type="primary"):
    if not resume_input.strip():
        st.warning("Please paste a resume first!")
    else:
        with st.spinner("Analyzing semantic context... 🤖"):
            query_clean = preprocess_text(resume_input)
            
            # --- STAGE 1: FAISS Retrieval ---
            bge_query = "Represent this sentence for searching relevant jobs: " + query_clean
            query_embedding = embedding_model.encode(
                [bge_query], 
                convert_to_numpy=True, 
                normalize_embeddings=True
            ).astype("float32")
            
            _, indices = faiss_index.search(query_embedding, 50)  # Cast a wide net of 50 jobs
            candidate_indices = indices[0].tolist()

            # --- STAGE 2: Cross-Encoder Reranking ---
            pairs = [[query_clean, d["combined_text"].iloc[idx]] for idx in candidate_indices]
            cross_scores = reranker.predict(pairs)
            
            scored_candidates = list(zip(candidate_indices, cross_scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # --- DISPLAY RESULTS ---
            st.subheader("Top Recommendations:")
            for rank, (idx, score) in enumerate(scored_candidates[:5]):
                job = d.iloc[idx]
                
                # Fetch Apply Link (Fallback to '#' if not available)
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
