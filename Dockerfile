# SciFlow Agent — Streamlit app image (CPU only, no GPU required)
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application (``.dockerignore`` excludes .env, .venv, caches).
COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD \
    python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
