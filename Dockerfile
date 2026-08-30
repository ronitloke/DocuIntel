FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        tesseract-ocr \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
# The container does not select host GPU hardware. Use the CPU PyTorch wheel
# so the Linux image does not pull an unnecessary CUDA runtime; the existing
# embedding and CrossEncoder models remain unchanged and load lazily.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        "torch==2.9.1+cpu" \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
COPY evaluation ./evaluation
COPY scripts ./scripts
COPY streamlit_app ./streamlit_app
COPY .streamlit ./.streamlit

RUN useradd --create-home --shell /usr/sbin/nologin docuintel \
    && mkdir -p /app/data/processed/uploads /app/data/processed/redacted /app/logs \
        /home/docuintel/.cache/huggingface /home/docuintel/.cache/sentence_transformers \
    && chown -R docuintel:docuintel /app /home/docuintel
USER docuintel

ENV HOME=/home/docuintel \
    HF_HOME=/home/docuintel/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/docuintel/.cache/sentence_transformers \
    TRANSFORMERS_CACHE=/home/docuintel/.cache/huggingface

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
