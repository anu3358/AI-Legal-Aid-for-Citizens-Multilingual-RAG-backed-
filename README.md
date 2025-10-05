# ⚖️ Nyāy Buddy — AI Legal Aid Assistant
(( POWERED BY ->RAG))
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
🏠 Tenancy / Housing

English: My landlord is refusing to return my security deposit. What should I do?

Hindi: मकान मालिक मेरी जमा राशि वापस नहीं कर रहा है। मुझे क्या करना चाहिए?

Punjabi: ਮੇਰਾ ਮਾਲਕ ਮੇਰੀ ਡਿਪਾਜ਼ਿਟ ਵਾਪਸ ਨਹੀਂ ਕਰ ਰਿਹਾ। ਮੈਂ ਕੀ ਕਰਾਂ?

🛒 Consumer Rights

English: I received a defective mobile phone. How can I get a refund?

Hindi: मुझे खराब मोबाइल फोन मिला है। मुझे रिफंड कैसे मिलेगा?

Punjabi: ਮੈਨੂੰ ਖਰਾਬ ਮੋਬਾਈਲ ਫੋਨ ਮਿਲਿਆ ਹੈ। ਮੈਂ ਰਿਫੰਡ ਕਿਵੇਂ ਲਵਾਂ?

💰 Salary & Labour

English: My employer has not paid me salary for two months. Where can I complain?

Hindi: मेरा नियोक्ता दो महीने से वेतन नहीं दे रहा है। मैं शिकायत कहाँ करूं?

Punjabi: ਮੇਰਾ ਨੌਕਰੀਦਾਤਾ ਦੋ ਮਹੀਨਿਆਂ ਤੋਂ ਤਨਖਾਹ ਨਹੀਂ ਦੇ ਰਿਹਾ। ਮੈਂ ਸ਼ਿਕਾਇਤ ਕਿੱਥੇ ਕਰਾਂ?

🖥️ Cybercrime

English: Someone hacked my UPI account and stole money. What should I do?

Hindi: किसी ने मेरा UPI अकाउंट हैक कर लिया और पैसे चुरा लिए। मुझे क्या करना चाहिए?

Punjabi: ਕਿਸੇ ਨੇ ਮੇਰਾ UPI ਖਾਤਾ ਹੈਕ ਕਰ ਲਿਆ ਅਤੇ ਪੈਸੇ ਚੋਰੀ ਕਰ ਲਏ। ਮੈਂ ਕੀ ਕਰਾਂ?

👩‍🦰 Women Protection

English: My husband beats me and threatens me. What legal protection can I get?

Hindi: मेरा पति मुझे मारता और धमकाता है। मुझे क्या कानूनी सुरक्षा मिल सकती है?

Punjabi: ਮੇਰਾ ਪਤੀ ਮੈਨੂੰ ਮਾਰਦਾ ਤੇ ਧਮਕਾਉਂਦਾ ਹੈ। ਮੈਨੂੰ ਕੀ ਕਾਨੂੰਨੀ ਸੁਰੱਖਿਆ ਮਿਲ ਸਕਦੀ ਹੈ?

🎁 Dowry Harassment

English: My in-laws are harassing me for dowry. What steps can I take?

Hindi: मेरे ससुराल वाले दहेज के लिए परेशान कर रहे हैं। मुझे क्या करना चाहिए?

Punjabi: ਮੇਰੇ ਸੱਸੁਰਾਲ ਵਾਲੇ ਮੈਨੂੰ ਦਾਜ਼ ਲਈ ਤੰਗ ਕਰ ਰਹੇ ਹਨ। ਮੈਂ ਕੀ ਕਰਾਂ?

👩‍⚖️ Caste Discrimination

English: How can I complain if someone insults me publicly because of my caste?

Hindi: अगर कोई मुझे जाति के कारण सार्वजनिक रूप से अपमानित करता है तो मैं शिकायत कैसे करूं?

Punjabi: ਜੇ ਕੋਈ ਮੈਨੂੰ ਜਾਤ ਦੇ ਆਧਾਰ 'ਤੇ ਸਰਵਜਨਿਕ ਰੂਪ ਵਿੱਚ ਬੇਇਜ਼ਤ ਕਰੇ ਤਾਂ ਮੈਂ ਸ਼ਿਕਾਇਤ ਕਿਵੇਂ ਕਰਾਂ?

📑 RTI

English: How do I file an RTI application?

Hindi: RTI आवेदन कैसे दाखिल करूं?

Punjabi: RTI ਅਰਜ਼ੀ ਕਿਵੇਂ ਦਵਾਂ?

⚖️ Lok Adalat

English: Can a cheque bounce case be settled in Lok Adalat?

Hindi: क्या चेक बाउंस का मामला लोक अदालत में निपटाया जा सकता है?

Punjabi: ਕੀ ਚੈਕ ਬਾਊਂਸ ਦਾ ਕੇਸ ਲੋਕ ਅਦਾਲਤ ਵਿੱਚ ਨਿਪਟਾਇਆ ਜਾ ਸਕਦਾ ਹੈ?

👶 Child Labour

English: I saw a child working at a factory. How can I report this?

Hindi: मैंने एक बच्चे को फैक्ट्री में काम करते देखा। मैं इसकी शिकायत कैसे करूं?

Punjabi: ਮੈਂ ਇੱਕ ਬੱਚੇ ਨੂੰ ਫੈਕਟਰੀ ਵਿੱਚ ਕੰਮ ਕਰਦਾ ਦੇਖਿਆ। ਮੈਂ ਇਸ ਦੀ ਸ਼ਿਕਾਇਤ ਕਿਵੇਂ ਕਰਾਂ?

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
