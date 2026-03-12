FROM python:3.11-slim

WORKDIR /app

# shapely needs libgeos; matplotlib runs in Agg (no display needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Layer-caching trick ────────────────────────────────────────────────────
# Install dependencies before copying source so this layer is re-used on
# every rebuild unless pyproject.toml actually changes.
COPY pyproject.toml .
RUN pip install --no-cache-dir \
        "shapely>=2.0" "numpy>=1.26" "ezdxf>=1.1" \
        "openpyxl>=3.1" "matplotlib>=3.8" "streamlit>=1.32"

# ── Application code ───────────────────────────────────────────────────────
COPY src/ src/
COPY data/ data/

# Install the package itself (no deps — already installed above)
RUN pip install --no-cache-dir --no-deps -e .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

ENTRYPOINT ["streamlit", "run", "src/layout_agent/streamlit_app.py", \
            "--server.port=8501", "--server.address=0.0.0.0", \
            "--server.headless=true"]
