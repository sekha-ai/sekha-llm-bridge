# Multi-stage build for sekha-llm-bridge
FROM python:3.14-slim AS builder

# OCI labels for GitHub Container Registry
LABEL org.opencontainers.image.source=https://github.com/sekha-ai/sekha-llm-bridge
LABEL org.opencontainers.image.description="Multi-provider LLM operations service with intelligent routing"
LABEL org.opencontainers.image.licenses=AGPL-3.0

WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry==1.7.1

# Configure Poetry
RUN poetry config virtualenvs.create false

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --only main --no-interaction --no-ansi --no-root

# Copy source code
COPY . .

# Install the project
RUN poetry install --only-root --no-interaction --no-ansi

# Runtime stage
FROM python:3.14-slim

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy installed packages
COPY --from=builder /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /app /app

# Create non-root user
RUN useradd -m -u 1000 sekha && chown -R sekha:sekha /app
USER sekha

# Set Python path to include src
ENV PYTHONPATH=/app/src

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:5001/health || exit 1

# Use the correct module path
CMD ["uvicorn", "sekha_llm_bridge.main:app", "--host", "0.0.0.0", "--port", "5001"]
