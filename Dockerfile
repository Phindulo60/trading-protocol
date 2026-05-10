FROM python:3.13-slim AS builder

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock ./

# Install deps into a virtual env
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY fsp/ ./fsp/
COPY README.md ./

# Install the project itself
RUN uv sync --frozen --no-dev

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.13-slim

WORKDIR /app

# Copy the virtual env from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source
COPY --from=builder /app/fsp /app/fsp
COPY --from=builder /app/pyproject.toml /app/

# Ensure .venv/bin is on PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create data directory for journal DB + cache
RUN mkdir -p /data/.fsp

# Config will be mounted or passed via env vars
ENV FSP_DATA_DIR=/data/.fsp

# Health check — process alive
HEALTHCHECK --interval=60s --timeout=5s --retries=5 --start-period=120s \
    CMD pgrep -f python || exit 1

ENTRYPOINT ["fsp"]
CMD ["live", "--feed", "td", "--llm", "--pairs", "EURUSD,GBPUSD,AUDUSD,USDCAD,EURJPY,GBPJPY", "--interval", "300"]
