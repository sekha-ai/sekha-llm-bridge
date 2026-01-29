# Multi-stage build for sekha-llm-bridge
FROM python:3.13-slim AS builder

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
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy entire Python installation from builder (includes interpreter, libraries, site-packages)
COPY --from=builder /usr/local /usr/local

# Copy application code
COPY --from=builder /app /app

# Create non-root user
RUN useradd -m -u 1000 sekha && chown -R sekha:sekha /app
USER sekha

# Set Python path to include src
ENV PYTHONPATH=/app/src
ENV PATH=/usr/local/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:5001/health || exit 1

# Use full Python path with -m uvicorn
CMD ["/usr/local/bin/python3.13", "-m", "uvicorn", "sekha_llm_bridge.main:app", "--host", "0.0.0.0", "--port", "5001"]
