# Build the package wheel.
FROM python:3.11-slim AS builder

# Install system build tools.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install the PEP 517 build toolchain.
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir build hatchling

WORKDIR /build

# Copy package metadata and source.
COPY pyproject.toml README.md LICENSE ./
COPY synkit/ ./synkit

RUN python -m build --wheel --no-isolation

# Create the runtime image.
FROM python:3.11-slim

WORKDIR /opt/synkit

COPY --from=builder /build/dist/*.whl ./

# Install the wheel and discard the build artifact.
RUN pip install --no-cache-dir *.whl \
    && rm *.whl

# Print the installed version by default.
CMD ["python", "-c", "import importlib.metadata as m; print(m.version('synkit'))"]
