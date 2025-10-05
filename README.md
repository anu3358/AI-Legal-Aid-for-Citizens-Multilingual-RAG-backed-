# Nyāy Buddy — Final Project Package

## Files
- streamlit_app_text_only.py  -> lightweight text-only demo (RAG + LLM)
- streamlit_app_full.py      -> full demo with audio upload + TTS
- kb.json                    -> knowledge base (your uploaded legal snippets)
- requirements.txt           -> Python dependencies
- README.md                  -> this file

## How to run locally / Colab
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the text-only demo (recommended for quick demo):
   ```bash
   streamlit run streamlit_app_text_only.py
   ```
3. Or run the full demo (supports audio upload):
   ```bash
   streamlit run streamlit_app_full.py
   ```
4. If running in Colab, use ngrok (pyngrok) to expose the streamlit port for a public link.

## Deploy to Streamlit Cloud
- Push this repo to GitHub and create a new Streamlit Cloud app pointing to either of the main files (streamlit_app_full.py or streamlit_app_text_only.py).

## Notes
- The FLAN-T5-small generator is used for demo; quality is moderate. Replace generator with API-backed LLM for production quality.
- Audio transcription (Whisper) may not be available on Streamlit Cloud; use Text mode if ASR fails.
- This project is educational and informational only — not legal advice.
