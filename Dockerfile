FROM python:3.12-slim

# Use the same resolver as local development; copying the binary avoids a
# separate pip bootstrap whose version could drift from the project's workflow.
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
# Resolve the locked production dependencies before copying application source
# so Docker can reuse this layer when only ATLAS code changes.
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

EXPOSE 8000
CMD ["/app/.venv/bin/atlas"]
