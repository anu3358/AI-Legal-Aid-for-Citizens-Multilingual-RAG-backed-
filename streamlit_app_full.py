import streamlit as st
import faiss
import torch
import textwrap
import json
import os
from transformers import pipeline
from deep_translator import GoogleTranslator
from gtts import gTTS

# ---------------------------
# Load Knowledge Base
# ---------------------------
@st.cache_resource
def load_kb(kb_file="kb.json"):
    with open(kb_file, "r", encoding="utf-8") as f:
        return json.load(f)

KB = load_kb()

# ---------------------------
# Embedding + FAISS Index
# ---------------------------
from sentence_transformers import SentenceTransformer

@st.cache_resource
def build_index(texts, embed_model):
    embeddings = embed_model.encode(texts, convert_to_tensor=False, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [d["text"] for d in KB]
index, kb_embeddings = build_index(texts, embed_model)

# ---------------------------
# QA Model
# ---------------------------
@st.cache_resource
def load_generator():
    return pipeline("text2text-generation", model="google/flan-t5-small")

generator = load_generator()

# ---------------------------
# Cleaner for Answers
# ---------------------------
def clean_answer(ans: str) -> str:
    """
    Clean model output:
    - Remove duplicate lines
    - Remove duplicate step blocks
    - Keep only one 'Example:' section
    """
    parts = [p.strip() for p in ans.split("\n") if p.strip()]
    cleaned = []
    seen = set()
    example_seen = False

    for p in parts:
        norm = p.lower().replace("  ", " ")
        if norm.startswith("example:"):
            if example_seen:
                continue
            example_seen = True
        if norm not in seen:
            cleaned.append(p)
            seen.add(norm)

    return "\n".join(cleaned)

# ---------------------------
# Answer Generation
# ---------------------------
def generate_answer(query: str, contexts: list, generator, max_len: int = 400) -> str:
    """Generate structured legal answers with repetition control."""
    contexts_txt = "\n".join([c["text"] for c in contexts])

    prompt = textwrap.dedent(f"""
    You are Nyāy Buddy, an AI assistant that explains Indian legal rights in plain, simple language.

    Context documents:
    {contexts_txt}

    User question: {query}

    Answer instructions:
    1. Explain the law in 2–3 sentences.
    2. Provide a single clear step-by-step list (only once, no repetition).
    3. Mention relevant helpline numbers.
    4. Add at most one example case if available.
    5. If unsure, say 'Please consult a lawyer or legal aid service.'
    Answer in full sentences, not one-liners.
    """)

    raw_out = generator(
        prompt,
        max_length=max_len,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.2
    )[0]["generated_text"].strip()

    return clean_answer(raw_out)

# ---------------------------
# Translator
# ---------------------------
def translate_text(text, src="auto", tgt="en"):
    try:
        return GoogleTranslator(source=src, target=tgt).translate(text)
    except:
        return text

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="⚖️ Nyāy Buddy", page_icon="⚖️", layout="wide")

st.title("⚖️ Nyāy Buddy — AI Legal Aid for Citizens")
st.write("Ask legal questions in Hindi, Punjabi, or English. Get plain answers + helpline numbers.")

mode = st.radio("Mode", ["Text", "Audio upload"])

if mode == "Text":
    user_q = st.text_input("Type your legal question (Hindi/Punjabi/English)")

    if user_q:
        # Translate to English for processing
        q_en = translate_text(user_q, src="auto", tgt="en")

        # Search in KB
        q_emb = embed_model.encode([q_en], convert_to_tensor=False, normalize_embeddings=True)
        scores, idxs = index.search(q_emb, 2)  # top-2 matches
        retrieved = [KB[i] for i in idxs[0]]

        # Generate Answer
        answer = generate_answer(q_en, retrieved, generator)

        # Translate back if needed
        answer_out = translate_text(answer, src="en", tgt="auto")

        # Show sources
        st.subheader("🔎 Sources retrieved")
        for d in retrieved:
            st.markdown(f"**{d['title']}** — {d['source']}")

        st.subheader("🗨️ Nyāy Buddy Answer")
        st.write(answer_out)

        # Optional TTS
        if st.checkbox("🔊 Listen Answer (gTTS)"):
            try:
                tts = gTTS(answer_out, lang="hi")  # auto lang detection may fail, forcing Hindi
                tts.save("answer.mp3")
                audio_file = open("answer.mp3", "rb")
                st.audio(audio_file.read(), format="audio/mp3")
            except:
                st.error("⚠️ TTS Error. Please try text only mode.")

elif mode == "Audio upload":
    st.info("🎤 Upload audio questions not enabled in this demo.")
