# Builder stage - Build environment for the application
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libgomp1 \
    gfortran \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/.venv
ENV PATH="/opt/.venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu \
    curl \
    ca-certificates \
    openssl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/.venv /opt/.venv
ENV PATH="/opt/.venv/bin:$PATH"
ENV STREAMLIT_TELEMETRY="0"

WORKDIR /app
COPY . .

EXPOSE 8080

CMD ["streamlit", "run", "app/Home.py", "--server.port=8080", "--server.address=0.0.0.0"]