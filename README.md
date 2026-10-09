# 📄 ATS Resume Checker

A Streamlit app that scores a resume for ATS (Applicant Tracking System) friendliness and gives concrete, prioritized improvements, powered by Google Gemini Flash.

## Features
- Upload a resume as **PDF, DOCX or TXT**
- Optional **job description** input for targeted keyword matching
- Overall **ATS score (0-100)** plus a 5-category breakdown
- Strengths, missing keywords, prioritized fixes, and before/after bullet rewrites
- Download the report as JSON

> The score is an AI-generated estimate, not the output of a real ATS.

## Project structure
```
.
├── app.py             # Streamlit app
├── requirements.txt   # Python dependencies
└── README.md
```

## Run locally
1. Get a free Gemini API key: https://aistudio.google.com/apikey
2. Install and run:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Provide the key in one of these ways:
   - Paste it into the sidebar (easiest), or
   - Set an environment variable: `export GEMINI_API_KEY="your-key"` (Windows PowerShell: `$env:GEMINI_API_KEY="your-key"`), or
   - Create `.streamlit/secrets.toml` containing `GEMINI_API_KEY = "your-key"`

## Choosing the model
The default is `gemini-flash-latest` (always points to the current Flash model). You can change it in the sidebar, or set the `GEMINI_MODEL` environment variable. Google retires old model IDs periodically, so if you see a "model not found" error, check https://ai.google.dev/gemini-api/docs/models for the current name.

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (never commit your API key).
2. Go to https://share.streamlit.io, sign in with GitHub and click **Create app**.
3. Select the repo, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Click **Deploy**.

## Privacy
Resume text is sent to Google's Gemini API for analysis and is not stored by this app.

## License
MIT
