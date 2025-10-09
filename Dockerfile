# ===== Builder stage =====
FROM python:3.13-slim-bookworm AS builder

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libgomp1 \
    gfortran \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
ENV ENV_PATH="/opt/.venv"
RUN python3 -m venv "${ENV_PATH}"
# Add virtual environment into PYTHON path
ENV PATH="${ENV_PATH}/bin:$PATH"

# Copy requirement file and install dependencies
## Gradio
COPY requirements.gradio.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.gradio.txt
## Streamlit
# COPY requirements.txt ./
# RUN pip install --no-cache-dir --upgrade pip \
#     && pip install --no-cache-dir -r requirements.txt

# Cleanup
RUN find "${ENV_PATH}" -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true \
    && find "${ENV_PATH}" -type d -name "test" -exec rm -rf {} + 2>/dev/null || true \
    && find "${ENV_PATH}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find "${ENV_PATH}" -type f -name "*.pyc" -delete \
    && find "${ENV_PATH}" -type f -name "*.pyo" -delete \
    && find "${ENV_PATH}" -type d -name "docs" -exec rm -rf {} + 2>/dev/null || true \
    && find "${ENV_PATH}" -type d -name "examples" -exec rm -rf {} + 2>/dev/null || true \
    && find "${ENV_PATH}" -type d -name "benchmarks" -exec rm -rf {} + 2>/dev/null || true

# ===== Runtime stage =====
FROM python:3.13-slim-bookworm AS runtime

# Create environment variable
ENV ENV_PATH="/opt/.venv"

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Cleanup
RUN rm -rf ~/.local ~/.cache

COPY --from=builder "${ENV_PATH}" "${ENV_PATH}"
# Add virtual environment into PYTHON path
ENV PATH="${ENV_PATH}/bin:$PATH"

# Remove comment if you want to run Streamlit application instead of Gradio application
# ENV STREAMLIT_TELEMETRY="0"

WORKDIR /app
COPY . .

## ==== Streamlit application ====
# EXPOSE 8080
# CMD ["streamlit", "run", "streamlit_app/Home.py", "--server.port=6000", "--server.address=0.0.0.0"]

## ==== Gradio application ====
EXPOSE 7000
CMD ["python", "./gradio_app/app.py"]