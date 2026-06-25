FROM python:3.12-slim

# Copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency definition, README, and source code
COPY pyproject.toml README.md /app/
COPY app/ /app/app/

# Install project dependencies globally
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system .

# Copy database migrations configuration
COPY migrations/ /app/migrations/
COPY alembic.ini /app/alembic.ini

# Expose FastAPI default port
EXPOSE 8000

# Enable unbuffered logging
ENV PYTHONUNBUFFERED=1

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
