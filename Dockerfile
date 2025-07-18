############################################
# STAGE 1: Build your package wheel
############################################
FROM python:3.11-slim AS builder

# 1. Install system build tools (for any C extensions)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip/setuptools/wheel and install build frontends
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir poetry-core build

WORKDIR /build

# 3. Copy pyproject.toml (and lockfile, if you have one)
COPY pyproject.toml ./
# COPY poetry.lock ./             # uncomment if you use Poetry lockfile

# 4. Copy package sources
COPY synkit/ ./synkit

# 5. Build the wheel
RUN python -m build --wheel --no-isolation
