# streamlit_app_text_only.py
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from googletrans import Translator
import json, os, textwrap

st.set_page_config(page_title="Nyāy Buddy — Legal Aid (Text Demo)", layout="wide",
                   initial_sidebar_state="expanded")

# --- Styling ---
st.markdown("""
<style>
.header {background: linear-gradient(90deg,#0f172a,#0f2438); padding:18px; border-radius:8px;}
.header h1{color: #fff; margin:0; font-size:30px;}
.card {background:#ffffff; padding:12px; border-radius:8px; box-shadow: 0 2px 6px rgba(15,23,42,0.08);}
.source {font-size:12px; color:#444; background:#f3f4f6; padding:8px; border-radius:6px;}
.user-bubble {background:#e6f4ea; padding:10px; border-radius:10px; display:inline-block;}
.bot-bubble {background:#f1f5f9; padding:10px; border-radius:10px; display:inline-block;}
.small {font-size:13px; color:#666;}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="header"><h1>⚖️ Nyāy Buddy — Legal Aid Assistant (Text Demo)</h1></div>', unsafe_allow_html=True)
st.write("")

# --- Sidebar ---
with st.sidebar.container():
    st.markdown("### Project Info")
    st.markdown("- B.Tech CSE Project\n- Punjabi University, Patiala\n- Multilingual RAG-backed legal assistant")
    st.markdown("### Deployment")
    st.markdown("- Streamlit Cloud: Text-only demo (fast & stable)")
    st.markdown("- Hugging Face Spaces: Full demo (voice + ASR)")
    st.markdown("### Tips")
    st.markdown("Use short, clear questions. Example: *Mera landlord deposit wapas nahi kar raha*")

# --- Load KB ---
KB_PATH = "kb.json"
if not os.path.exists(KB_PATH):
    st.error("kb.json not found in project folder. Please add your KB JSON file named kb.json.")
    st.stop()

with open(KB_PATH, "r", encoding="utf-8") as f:
    KB = json.load(f)

# --- Models (cached) ---
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def build_index(kb, embed_model):
    texts = [d["text"] for d in kb]
    embs = embed_model.encode(texts, convert_to_numpy=True)
    d = embs.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embs)
    return index, embs

@st.cache_resource(show_spinner=False)
def get_generator():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
    return gen

embed_model = get_embedding_model()
index, kb_embeddings = build_index(KB, embed_model)
generator = get_generator()
translator = Translator()

# --- Retrieval & Generation ---
def retrieve(query, k=3):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    results = []
    for idx in I[0]:
        if idx < len(KB):
            results.append(KB[idx])
    return results

def generate_answer(query, contexts, user_lang="en"):
    contexts_txt = "\n\n".join([f"Source: {c['source']}\nText: {c['text']}" for c in contexts])
    prompt = textwrap.dedent(f"""You are an assistant that explains Indian legal texts in simple, short sentences for a common person.
    Context documents:
    {contexts_txt}

    User question: {query}

    Task: Give a short plain-language answer (2-6 sentences), list step-by-step actions the user can take, and mention the source titles used. If unsure, say 'Please consult a lawyer or legal aid'.
    """)
    out = generator(prompt, max_length=200, do_sample=False)[0]["generated_text"]
    if user_lang != "en":
        try:
            return translator.translate(out, dest=user_lang).text
        except Exception:
            return out
    return out

# --- Chat-like UI ---
st.markdown("## Ask Nyāy Buddy")
with st.form("ask"):
    q = st.text_area("Type your question (Hindi / Punjabi / English)", height=120)
    submitted = st.form_submit_button("Ask")
if submitted and q.strip():
    with st.spinner("Thinking..."):
        # detect language & translate for retrieval
        try:
            detected = translator.detect(q)
            user_lang = detected.lang
        except Exception:
            user_lang = "en"
        q_en = q if user_lang == "en" else translator.translate(q, dest="en").text

        # retrieve and show sources
        docs = retrieve(q_en, k=3)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🔎 Sources retrieved")
        for d in docs:
            st.markdown(f"**{d['title']}** — <span class='small'>{d['source']}</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='source'>{d['text']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # generate answer
        ans = generate_answer(q_en, docs, user_lang=user_lang)
        st.markdown("<br/>")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🗨️ Nyāy Buddy (Simple Answer)")
        st.markdown(f"<div class='bot-bubble'>{ans}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.info("Disclaimer: This is informational only. Consult a qualified lawyer for representation.")
