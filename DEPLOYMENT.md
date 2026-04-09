# Deployment Notes

This project is a Streamlit app. It is ready for Git-based deployment on hosts that run long-lived Python web processes, such as Streamlit Community Cloud, Render, Railway, or a VPS.

## Recommended settings

Use these settings on a Python app host:

- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
- Python version: `3.12`

## Required environment variables

Add these in the hosting dashboard if you use the related features:

- `FIREBASE_API_KEY`
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_REDIRECT_URI`
- `GEMINI_API_KEY`

Do not commit `.streamlit/secrets.toml`; it is already ignored.

## Vercel status

Vercel's Python runtime is for ASGI/WSGI apps such as FastAPI and Flask. This app is Streamlit, which runs as a long-lived Tornado/WebSocket server, so it is not a good fit for Vercel's Python Functions.

The repository also ignores the large local training/data files:

- `mandi_master_data.parquet`
- `MinorP Dataset/`
- `*.pth`
- `*.npz`

That keeps GitHub and deployment bundles small, but the deployed app will use the live API and `mini_fallback.csv` unless you move the large data/model artifacts to external storage or a database.
