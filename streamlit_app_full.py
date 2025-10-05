# streamlit_app_full.py
# Nyāy Buddy — Full Streamlit demo (Text + Audio upload, RAG + LLM, TTS)
# Usage: streamlit run streamlit_app_full.py

import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from googletrans import Translator
from gtts import gTTS
import tempfile, os, uuid, json

st.set_page_config(page_title="Nyāy Buddy — Full Demo", layout="centered")

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

def tts_play(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tf:
            tts.save(tf.name)
            tf.flush()
            with open(tf.name, 'rb') as f:
                data = f.read()
        os.remove(tf.name)
        return data
    except Exception as e:
        st.warning('TTS failed: ' + str(e))
        return None

def load_kb(path="kb.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def retrieve_top_k(query, k=3):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    results = []
    for idx in I[0]:
        if idx < len(KB):
            results.append(KB[idx])
    return results

def generate_answer(query, context_passages, user_lang='en'):
    contexts = "\\n\\n".join([f"Source: {p['source']}\\nText: {p['text']}" for p in context_passages])
    prompt = (
        "You are an assistant that explains Indian legal texts in simple language for a common person.\\n\\n"
        f"Context documents:\\n{contexts}\\n\\n"
        f"User question: {query}\\n\\n"
        "Task: Give a short plain-language answer (2-6 sentences) summarizing the relevant law, "
        "what actions the user can take (step-by-step), and cite the source titles. "
        "If unsure, say 'Please consult a lawyer or legal aid'. Keep answer concise."
    )
    out = generator(prompt, max_length=256, do_sample=False)[0]['generated_text']
    if user_lang != 'en':
        try:
            trans = translator.translate(out, dest=user_lang).text
            return trans
        except Exception:
            return out
    return out

# Initialize models
KB = load_kb(KB_PATH)
embed_model = load_embedding_model()
index, kb_embeddings = build_faiss_index(KB, embed_model)
generator = load_generator()
translator = Translator()

st.title("Nyāy Buddy — Full Demo (Text + Audio)")
st.markdown("Ask in text or upload a short audio file (wav/mp3). The system will transcribe and answer based on the KB.")

mode = st.radio("Mode", ("Text", "Audio upload"))

if mode == "Text":
    with st.form("form_text"):
        q = st.text_area("Type your legal question:", height=140)
        submitted = st.form_submit_button("Ask")
    if submitted and q.strip():
        with st.spinner("Processing..."):
            try:
                detected = translator.detect(q)
                user_lang = detected.lang
            except Exception:
                user_lang = 'en'
            if user_lang != 'en':
                try:
                    q_en = translator.translate(q, dest='en').text
                except:
                    q_en = q
            else:
                q_en = q
            top_docs = retrieve_top_k(q_en, k=3)
            st.subheader("Sources retrieved:")
            for d in top_docs:
                st.write(f\"**{d['title']}** — {d['source']}\")
                st.caption(d['text'])
            answer = generate_answer(q_en, top_docs, user_lang=user_lang)
            st.subheader("Nyāy Buddy Answer:")
            st.write(answer)
            tts_lang = 'hi' if user_lang.startswith('hi') else ('pa' if user_lang.startswith('pa') else 'en')
            audio_bytes = tts_play(answer, lang=tts_lang)
            if audio_bytes:
                st.audio(audio_bytes, format='audio/mp3')
            st.info('Disclaimer: Informational only. Consult a lawyer.')
else:
    st.info('Upload audio (wav/mp3). We try to transcribe using Whisper (if available).')
    audio_file = st.file_uploader('Upload audio file', type=['wav','mp3','m4a','ogg'])
    if audio_file is not None:
        with st.spinner('Transcribing...'):
            tmp_path = os.path.join(tempfile.gettempdir(), f\"upload_{uuid.uuid4().hex}.wav\")
            with open(tmp_path, 'wb') as f:
                f.write(audio_file.read())
            try:
                from transformers import pipeline as hf_pipeline
                asr = hf_pipeline('automatic-speech-recognition', model='openai/whisper-small')
                asr_out = asr(tmp_path)
                query_text = asr_out.get('text', '')
            except Exception as e:
                st.warning('Transcription unavailable: please use Text mode.')
                query_text = ''
            if query_text:
                st.write('Transcribed text:')
                st.write(query_text)
                try:
                    detected = translator.detect(query_text)
                    user_lang = detected.lang
                except Exception:
                    user_lang = 'en'
                if user_lang != 'en':
                    try:
                        q_en = translator.translate(query_text, dest='en').text
                    except:
                        q_en = query_text
                else:
                    q_en = query_text
                top_docs = retrieve_top_k(q_en, k=3)
                st.subheader('Sources retrieved:')
                for d in top_docs:
                    st.write(f\"**{d['title']}** — {d['source']}\")
                    st.caption(d['text'])
                answer = generate_answer(q_en, top_docs, user_lang=user_lang)
                st.subheader('Nyāy Buddy Answer:')
                st.write(answer)
                tts_lang = 'hi' if user_lang.startswith('hi') else ('pa' if user_lang.startswith('pa') else 'en')
                audio_bytes = tts_play(answer, lang=tts_lang)
                if audio_bytes:
                    st.audio(audio_bytes, format='audio/mp3')
                st.info('Disclaimer: Informational only. Consult a lawyer.')
            else:
                st.error('Transcription failed. Please try again or switch to Text mode.')
