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

# --- Build index (embedder inside cache) ---
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
def retrieve(query, k=1):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    results = []
    for idx in I[0]:
        if idx < len(KB):
            results.append(KB[idx])
    return results

def generate_answer(query, contexts, user_lang="en"):
    contexts_txt = "\n\n".join([f"Source: {c['source']}\nText: {c['text']}" for c in contexts])
    prompt = textwrap.dedent(f"""
    You are Nyāy Buddy, an AI assistant that explains Indian legal rights in plain, simple language.

    Context documents (use ONLY these, do not mix with unrelated topics):
    {contexts_txt}

    User question: {query}

    Answer instructions:
    - Start with a short 2–3 sentence plain explanation of the law.
    - Then give a numbered step-by-step list of clear actions (at least 3 steps).
    - Mention helpline numbers or official websites if available.
    - If unsure, say 'Please consult a lawyer or legal aid service.'
    Ensure the answer is complete, not just "1." — always give full sentences.
    """)
    out = generator(
        prompt,
        max_length=350,
        min_length=80,
        do_sample=False
    )[0]["generated_text"]

    if user_lang != "en":
        try:
            return GoogleTranslator(source="en", target=user_lang).translate(out)
        except Exception:
            return out
    return out

def tts_bytes(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
            tts.save(tf.name)
            tf.flush()
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

            docs = retrieve(q_en, k=1)
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

else:
    st.info("Upload a short audio file (wav/mp3). We attempt transcription with Whisper (transformers) if available.")
    audio_file = st.file_uploader("Upload audio", type=['wav','mp3','m4a','ogg'])
    if audio_file is not None:
        with st.spinner("Transcribing audio..."):
            tmp = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4().hex}.wav")
            with open(tmp, "wb") as f:
                f.write(audio_file.read())
            query_text = ""
            try:
                from transformers import pipeline as hf_pipeline
                asr = hf_pipeline("automatic-speech-recognition", model="openai/whisper-small")
                asr_out = asr(tmp)
                query_text = asr_out.get("text", "")
            except Exception as e:
                st.warning("Whisper ASR unavailable: " + str(e))

            if query_text:
                st.write("Transcribed text:")
                st.write(query_text)

                try:
                    q_en = GoogleTranslator(source="auto", target="en").translate(query_text)
                except Exception:
                    q_en, user_lang = query_text, "en"

                user_lang = "en" if query_text == q_en else "hi" if any(c in query_text for c in "अआइईउऊएऐਓऔਕਖਗਘਚਛਜਝਟਠਡਢਤਥਦਧਨਪਫਬਭਮਯਰਲਵਸ਼ਸਹ") else "pa"

                docs = retrieve(q_en, k=1)
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
            else:
                st.error("Transcription failed. Try again or use Text mode.")
