# streamlit_app_full.py
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from deep_translator import GoogleTranslator
from gtts import gTTS
import tempfile, os, uuid, json, textwrap

st.set_page_config(page_title="Nyāy Buddy — Full Demo", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.header {background: linear-gradient(90deg,#053e57,#0f172a); padding:14px; border-radius:8px;}
.header h1{color: #fff; margin:0; font-size:26px;}
.card {background:#ffffff; padding:12px; border-radius:8px; box-shadow: 0 2px 6px rgba(15,23,42,0.06);}
.source {font-size:12px; color:#444; background:#f3f4f6; padding:8px; border-radius:6px;}
.bot-bubble {background:#f1f5f9; padding:10px; border-radius:10px; display:inline-block;}
small{font-size:13px; color:#666;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>⚖️ Nyāy Buddy — Full Demo (Text + Audio)</h1></div>', unsafe_allow_html=True)
st.write("")

# Load KB
KB_PATH = "kb.json"
if not os.path.exists(KB_PATH):
    st.error("kb.json not found. Please upload your KB file named kb.json in the project root.")
    st.stop()

with open(KB_PATH, "r", encoding="utf-8") as f:
    KB = json.load(f)

# --- Generator (cached) ---
@st.cache_resource(show_spinner=False)
def get_generator():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
    return gen

generator = get_generator()

# --- Build index ---
@st.cache_resource(show_spinner=False)
def build_index(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(texts, convert_to_numpy=True)
    d = embs.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embs)
    return index, embs, model

texts = [d["text"] for d in KB]
index, kb_embeddings, embed_model = build_index(texts)

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

    try:
        out = generator(prompt, max_length=250, do_sample=False)[0]["generated_text"].strip()
    except Exception:
        out = ""

    if not out:
        out = "Please consult a lawyer or legal aid. I could not find a clear answer."

    if user_lang != "en":
        try:
            return GoogleTranslator(source="en", target=user_lang).translate(out)
        except Exception:
            return out
    return out

def tts_bytes(text, lang='en'):
    if not text or not text.strip():
        return None  # avoid sending empty text to gTTS
    try:
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
            tts.save(tf.name)
            with open(tf.name, "rb") as f:
                data = f.read()
        os.remove(tf.name)
        return data
    except Exception as e:
        st.warning("TTS error: " + str(e))
        return None

# --- UI ---
mode = st.radio("Mode", ("Text", "Audio upload"))

if mode == "Text":
    with st.form("text_form"):
        q = st.text_area("Type your legal question (Hindi/Punjabi/English)", height=140)
        submitted = st.form_submit_button("Ask Nyāy Buddy")
    if submitted and q.strip():
        with st.spinner("Processing..."):
            try:
                q_en = GoogleTranslator(source="auto", target="en").translate(q)
            except Exception:
                q_en, user_lang = q, "en"

            user_lang = "en" if q == q_en else "hi" if any(c in q for c in "अआइईउऊएऐओऔकखगघचछजझटठडढतथदधनपफबभमयरलवशषसह") else "pa"

            docs = retrieve(q_en, k=3)
            st.markdown('<div class="card">### 🔎 Sources retrieved</div>', unsafe_allow_html=True)
            for d in docs:
                st.markdown(f"**{d['title']}** — <small>{d['source']}</small>", unsafe_allow_html=True)
                st.markdown(f"<div class='source'>{d['text']}</div>", unsafe_allow_html=True)

            ans = generate_answer(q_en, docs, user_lang=user_lang)
            st.markdown('<div class="card"><h3>🗨️ Nyāy Buddy Answer</h3></div>', unsafe_allow_html=True)
            st.markdown(f"<div class='bot-bubble'>{ans}</div>", unsafe_allow_html=True)

            lang_code = 'hi' if user_lang == 'hi' else ('pa' if user_lang == 'pa' else 'en')
            audio = tts_bytes(ans, lang=lang_code)
            if audio:
                st.audio(audio, format='audio/mp3')

            st.info("Disclaimer: Informational only. Consult a qualified lawyer.")
