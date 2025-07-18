############################################
# STAGE 1: Build your package wheel
############################################
FROM python:3.11-slim AS builder

# 1. Set working dir
WORKDIR /build

# 2. Install PEP‑517 build frontend
RUN pip install --no-cache-dir build

# 3. Copy only pyproject.toml (and lockfile, if you have one) for caching
COPY pyproject.toml ./
# If you’re using Poetry or another lockfile, uncomment:
# COPY poetry.lock ./

# 4. Copy your package source folder
COPY synkit/ ./synkit

# 5. Build a wheel
RUN python -m build --wheel --no-isolation

############################################
# STAGE 2: Create the “release” image
############################################
FROM python:3.11-slim

WORKDIR /opt/synkit

# 6. Copy in the built wheel
COPY --from=builder /build/dist/*.whl ./

# 7. Install the wheel (and dependencies)
RUN pip install --no-cache-dir *.whl \
    && rm *.whl

# 8. (Optional) if your package defines console scripts in pyproject.toml, you can
#     set an ENTRYPOINT so users can call them directly:
# ENTRYPOINT ["synkit-cli"]
# CMD ["--help"]

# 9. Sanity check: print the installed synkit version via importlib.metadata
CMD ["python", "-c", "import importlib.metadata as m; print(m.version('synkit'))"]
