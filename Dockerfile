# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system build dependencies (e.g., gcc for C extensions if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for dependency isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Stage 2: Runtime image
FROM python:3.11-slim AS runner

# Metadata labels
LABEL maintainer="Maira Nawani" \
      org.opencontainers.image.title="AegisNet" \
      org.opencontainers.image.description="AI-Powered Network Intrusion Detection & Response Engine" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.authors="Maira Nawani"

WORKDIR /app

# Install runtime utilities (curl for container health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create dedicated non-root application user
RUN groupadd --system --gid 1001 aegisuser && \
    useradd --system --uid 1001 --gid aegisuser --create-home --home-dir /home/aegisuser aegisuser

# Ensure required runtime directories exist with appropriate permissions
RUN mkdir -p /app/models /app/data /app/logs /app/outputs && \
    chown -R aegisuser:aegisuser /app

# Copy application source code
COPY --chown=aegisuser:aegisuser . /app

# Configure runtime environment
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER aegisuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
