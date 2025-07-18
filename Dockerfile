############################################
# STAGE 1: Build your package wheel
############################################
FROM python:3.11-slim AS builder

# 1. Install system build tools (for any C extensions)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip/setuptools/wheel and install PEP 517 tooling + Hatchling backend
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir build hatchling

# 3. Set working directory inside builder
WORKDIR /build

# 4. Copy project metadata (including README so Hatchling can find it)
COPY pyproject.toml README.md ./
# If you have a lockfile, uncomment:
# COPY poetry.lock ./

# 5. Copy your library source
COPY synkit/ ./synkit

# 6. Build the wheel
RUN python -m build --wheel --no-isolation

############################################
# STAGE 2: Create the “release” image
############################################
FROM python:3.11-slim

# 7. Set a clean workdir
WORKDIR /opt/synkit

# 8. Copy in the built wheel from the builder stage
COPY --from=builder /build/dist/*.whl ./

# 9. Install your package (and its dependencies), then remove the wheel
RUN pip install --no-cache-dir *.whl \
    && rm *.whl

# 10. Sanity check: print the installed synkit version
CMD ["python", "-c", "import importlib.metadata as m; print(m.version('synkit'))"]
