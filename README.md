# ⚖️ Nyāy Buddy — AI Legal Aid Assistant

**Nyāy Buddy** is a multilingual AI-powered legal assistant that helps Indian citizens understand their legal rights in **simple language** (Hindi, Punjabi, English).  
It uses **Retrieval-Augmented Generation (RAG)** with legal documents + AI models, and provides **text and audio answers**.

---

## ✨ Features
- ✅ Multilingual: Hindi, Punjabi, English  
- ✅ Retrieval-Augmented Generation (RAG) with FAISS vector search  
- ✅ Simple explanations of legal rights and actions  
- ✅ Text-to-Speech (answers in user’s language)  
- ✅ Optional speech input (via Whisper ASR, Hugging Face Spaces)  
- ✅ Deployed on:
  - **Streamlit Cloud** → Text-only demo (fast & stable)  
  - **Hugging Face Spaces** → Full demo (voice + audio answers)  

---

## 🔗 Live Demo
Check out the live deployed app here:  
👉 [Nyāy Buddy Demo (Text-only)](https://nbewp2cytifnsycswqdnfi.streamlit.app/)

---


🏆 Author & Contributions

👤 ANURAG SAINI
B.Tech CSE, Punjabi University, Patiala

Specialized Work by Me:

📌 Conceptualization & Problem Identification → Identified the gap in citizen-friendly legal aid.

📌 Model Integration → Integrated RAG pipeline (FAISS + Sentence Transformers).

📌 Multilingual Support → Added Hindi, Punjabi, and English support using Deep Translator + TTS.

📌 Application Development → Designed and implemented the Streamlit apps (Text-only & Full demo).

📌 Deployment → Successfully deployed the app on Streamlit Cloud and prepared Hugging Face integration.

📌 Documentation & Reporting → Prepared technical documentation, README, and PPT material for the project.

## 🛠️ Project Structure



project/
│── kb.json # Knowledge base (legal texts)
│── streamlit_app_text_only.py # Streamlit app (text-only, deploy on Streamlit Cloud)
│── streamlit_app_full.py # Streamlit app (text + audio, deploy on Hugging Face Spaces)
│── requirements.txt # Python dependencies
│── README.md # Project documentation


---

## 🚀 Running Locally

### 1. Clone the repo
```bash
git clone <repo-url>
cd project


2. Install dependencies
pip install -r requirements.txt

3. Run Streamlit app
streamlit run streamlit_app_text_only.py


🌐 Deployment Options
1. Streamlit Cloud (Recommended for Demo)

Upload this repo to GitHub

Go to https://share.streamlit.io

Deploy directly from GitHub

App runs free with text-only mode

2. Hugging Face Spaces (Full Demo with Audio)

Create a new Space on Hugging Face

Select Streamlit as SDK

Upload all project files

App will support voice input + TTS answers

🧩 Example Queries

Try asking questions in Hindi, Punjabi, or English:

"मेरा मकान मालिक जमा वापस नहीं कर रहा" → Tenant rights in Hindi

"ਮੇਰਾ ਮਾਲਕ ਡਿਪਾਜ਼ਿਟ ਵਾਪਸ ਨਹੀਂ ਦੇ ਰਿਹਾ" → Tenant rights in Punjabi

"How to file RTI?" → Step-by-step Right to Information

📊 Architecture Overview

User Query (Text/Audio) → Input in Hindi, Punjabi, or English

Translation → Auto-translated to English (if needed) using Deep Translator

Vector Search (FAISS) → Relevant laws fetched from KB (kb.json)

LLM (Flan-T5) → Generates plain-language explanation & step-by-step actions

Answer Output → Text + (optional) Speech using gTTS

⚠️ Disclaimer

This tool is for educational and informational purposes only.
It is not a substitute for professional legal advice.
Always consult a qualified lawyer for legal matters.

👨‍💻 Project Info

Major Project (B.Tech CSE, 5th Semester)

University: Punjabi University, Patiala

Focus Areas: AI, ML, Deep Learning, RAG, LLMs, Neural Networks

Goal: Build a real problem-solving project for citizens



---

✅ Copy-paste this into `README.md` in your repo — it’s complete, clean, and formatted for GitHub.  

Do you also want me to create a **professional PPTX (PowerPoint)** version of this README for your **project presentation**?

## Notes
- The FLAN-T5-small generator is used for demo; quality is moderate. Replace generator with API-backed LLM for production quality.
- Audio transcription (Whisper) may not be available on Streamlit Cloud; use Text mode if ASR fails.
- This project is educational and informational only — not legal advice.
