import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer, CrossEncoder

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="AI Job Matcher", page_icon="🎯", layout="centered")

st.title("🎯 AI Job Matcher")
st.markdown("Powered by Two-Stage Semantic Retrieval (Bi-Encoder + Cross-Encoder)")

# ==========================================
# 1. CACHED MODEL LOADING
# ==========================================
# @st.cache_resource ensures the heavy AI models only load once when the server starts
@st.cache_resource(show_spinner="Loading AI Models... (This takes a minute on startup)")
def load_models():
    embedder = SentenceTransformer("intfloat/e5-base-v2")
    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return embedder, reranker_model

embedding_model, reranker = load_models()

# ==========================================
# 2. CACHED DATA & FAISS INDEX
# ==========================================
def preprocess_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

@st.cache_resource(show_spinner="Building Vector Search Index...")
def prepare_system():
    # Load the JSON file
    df = pd.read_json('job_dataset.json')
    
    # Extract lists into strings and build combined text
    def join_list(item):
        return ", ".join(item) if isinstance(item, list) else str(item)

    df["combined_text"] = (
        "Job Title: " + df.get("Title", "") + ". " +
        "Experience Level: " + df.get("ExperienceLevel", "") + ". " +
        "Skills: " + df.get("Skills", "").apply(join_list) + ". " +
        "Responsibilities: " + df.get("Responsibilities", "").apply(join_list)
    )

    df["combined_text"] = df["combined_text"].apply(preprocess_text)

    # Build FAISS Index
    job_embeddings = embedding_model.encode(
        df["combined_text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

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
            
            bge_query = "Represent this sentence for searching relevant jobs: " + query_clean
            query_embedding = embedding_model.encode(
                [bge_query], 
                convert_to_numpy=True, 
                normalize_embeddings=True
            ).astype("float32")
            
            _, indices = faiss_index.search(query_embedding, 50) 
            candidate_indices = indices[0].tolist()

            pairs = [[query_clean, d["combined_text"].iloc[idx]] for idx in candidate_indices]
            cross_scores = reranker.predict(pairs)
            
            scored_candidates = list(zip(candidate_indices, cross_scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            st.subheader("Top Recommendations:")
            for rank, (idx, score) in enumerate(scored_candidates[:5]):
                job = d.iloc[idx]
                
                with st.container():
                    st.markdown(f"""
                    <div style="border-left: 4px solid #4F46E5; background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 10px;">
                        <h4 style="margin:0; color: #1f2937;">{job.get('Title', 'Unknown Title')}</h4>
                        <p style="margin: 5px 0; color: #4b5563; font-size: 14px;">
                            💼 Experience: {job.get('ExperienceLevel', 'N/A')} ({job.get('YearsOfExperience', 'N/A')} years)
                        </p>
                        <span style="background-color: #D1FAE5; color: #065F46; padding: 3px 8px; border-radius: 10px; font-size: 12px; font-weight: bold;">
                            Relevance Score: {round(float(score) * 100, 2)}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)