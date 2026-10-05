"""
Entry point for Streamlit. Run from the repo root with:

    streamlit run main.py

Kept separate from app/frontend/streamlit_app.py so that the `app.*` absolute
imports used throughout the backend resolve correctly: Streamlit puts THIS
script's folder (the repo root, where the `app` package lives) on sys.path.
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "app" / "frontend" / "streamlit_app.py"), run_name="__main__")
