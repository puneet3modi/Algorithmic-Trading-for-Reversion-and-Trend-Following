FROM python:3.13-slim

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first for layer caching
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Default: show help
CMD ["bash", "-lc", "python --version && ls -la && echo 'Container ready. Use Makefile targets.'"]
