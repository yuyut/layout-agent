@echo off
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo Setting up virtual environment...
    py -3.9 -m venv .venv 2>nul || py -3 -m venv .venv 2>nul || python -m venv .venv
    .venv\Scripts\pip install -e ".[ui]"
)

echo Starting Facility Layout Optimiser...
start http://localhost:8501
.venv\Scripts\streamlit run src/layout_agent/streamlit_app.py
