#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -f .venv/bin/python ]; then
    echo "Setting up virtual environment..."
    python3 -m venv .venv
    .venv/bin/pip install -e ".[ui]"
fi

echo "Starting Facility Layout Optimiser..."
# Open browser after a short delay
(sleep 2 && open "http://localhost:8501" 2>/dev/null || xdg-open "http://localhost:8501" 2>/dev/null) &

.venv/bin/streamlit run src/layout_agent/streamlit_app.py
