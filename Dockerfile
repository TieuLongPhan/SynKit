FROM python:3.11-slim AS builder

# 1. (Optional) system deps for C extensions
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade core tooling and install PEP517 build frontend + Hatchling backend
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir build hatchling

WORKDIR /build

# 3. Copy only metadata for caching
COPY pyproject.toml ./
# COPY poetry.lock ./        # if you have one

# 4. Copy your library source
COPY synkit/ ./synkit

# 5. Build the wheel
RUN python -m build --wheel --no-isolation
