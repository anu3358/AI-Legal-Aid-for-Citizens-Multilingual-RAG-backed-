# streamlit_app_text_only.py
# Nyāy Buddy — Text-only Streamlit demo (RAG + LLM)
# Usage: streamlit run streamlit_app_text_only.py

import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from googletrans import Translator
import json, os

st.set_page_config(page_title="Nyāy Buddy — Text Demo", layout="centered")

# Load KB
KB_PATH = "kb.json"
with open(KB_PATH, "r", encoding="utf-8") as f:
    KB = json.load(f)

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def build_faiss_index(kb, embed_model):
    texts = [d["text"] for d in kb]
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index, embeddings

@st.cache_resource(show_spinner=False)
def load_generator():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
    return gen

# Initialize
embed_model = load_embedding_model()
index, kb_embeddings = build_faiss_index(KB, embed_model)
generator = load_generator()
translator = Translator()

def retrieve_top_k(query, k=3):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    results = []
    for idx in I[0]:
        if idx < len(KB):
            results.append(KB[idx])
    return results

def generate_answer(query, context_passages, user_lang="en"):
    contexts = "\\n\\n".join([f\"Source: {p['source']}\\nText: {p['text']}\" for p in context_passages])
    prompt = (
        "You are an assistant that explains Indian legal texts in simple language for a common person.\\n\\n"
        f"Context documents:\\n{contexts}\\n\\n"
        f"User question: {query}\\n\\n"
        "Task: Give a short plain-language answer (2-6 sentences) summarizing the relevant law, "
        "what actions the user can take (step-by-step), and cite the source titles. "
        "If unsure, say 'Please consult a lawyer or legal aid'. Keep answer concise."
    )
    out = generator(prompt, max_length=256, do_sample=False)[0]["generated_text"]
    if user_lang != "en":
        try:
            trans = translator.translate(out, dest=user_lang).text
            return trans
        except Exception:
            return out
    return out

# UI
st.title("Nyāy Buddy — Text-only Demo (RAG-backed)")
st.markdown("Type a legal question in Hindi/Punjabi/English. Responses are based on indexed legal snippets (KB).")

with st.form("query_form"):
    user_input = st.text_area("Enter your question:", height=140)
    submitted = st.form_submit_button("Ask Nyāy Buddy")

if submitted and user_input.strip():
    with st.spinner("Processing..."):
        try:
            detected = translator.detect(user_input)
            user_lang = detected.lang
        except Exception:
            user_lang = "en"
        if user_lang != "en":
            try:
                query_en = translator.translate(user_input, dest="en").text
            except Exception:
                query_en = user_input
        else:
            query_en = user_input

        top_docs = retrieve_top_k(query_en, k=3)
        st.subheader("Sources retrieved:")
        for d in top_docs:
            st.write(f\"**{d['title']}** — {d['source']}\")
            st.caption(d['text'])

        answer = generate_answer(query_en, top_docs, user_lang=user_lang)
        st.subheader(\"Nyāy Buddy Answer:\")
        st.write(answer)

        st.info(\"Disclaimer: Informational only. Consult a lawyer for representation.\")
else:
    st.write(\"Enter a question and press Ask Nyāy Buddy.\")
