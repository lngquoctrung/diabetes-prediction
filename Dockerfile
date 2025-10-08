# Builder stage
FROM python:3.13-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make libgomp1 gfortran \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.13-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    fonts-dejavu \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Remove comment if you want to run Streamlit application instead of Gradio application
# ENV STREAMLIT_TELEMETRY="0"

WORKDIR /app
COPY . .

# Cleanup
RUN find /usr/local/lib/python3.11/site-packages -type d -name "tests" -exec rm -rf {} + \
    && find /usr/local/lib/python3.11/site-packages -type d -name "__pycache__" -exec rm -rf {} + \
    && find /usr/local/lib/python3.11/site-packages -type f -name "*.pyc" -delete \
    && rm -rf /root/.cache /root/.local

## ==== Streamlit application ====
# EXPOSE 8080
# CMD ["streamlit", "run", "streamlit_app/Home.py", "--server.port=6000", "--server.address=0.0.0.0"]

## ==== Gradio application ====
EXPOSE 7000
CMD ["python", "./gradio_app/app.py"]
