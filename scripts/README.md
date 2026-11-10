# Repository scripts

The shell helpers work from any directory and use the active Python environment.
Set `PYTHON` to select another interpreter.

## Checks

```bash
bash scripts/lint.sh
bash scripts/pytest.sh
bash scripts/build_doc.sh
```

- `lint.sh` runs source-size, docstring and Flake8 checks.
- `pytest.sh` runs `Test/` and available experiment test suites. Arguments are
  passed to pytest, for example `bash scripts/pytest.sh Test/CRN -q`.
- `build_doc.sh` builds the Sphinx documentation into `docs/` and fails on warnings.

Native runtime tests share one build per component. Reuse verified builds
between local runs with:

```bash
SYNKIT_TEST_NATIVE_CACHE="$PWD/build/test-native-cache" bash scripts/pytest.sh
```

CI caches Conda downloads and these native outputs in addition to pip downloads.
Native cache entries depend on the OS, architecture, Python, runner image,
toolchain, dependencies and sources. Builders verify restored configurations,
library hashes and ABIs. Build-integrity and concurrency tests keep fresh
directories. Completed builds are saved even if a test fails; cancelled runs
are not saved. A new push cancels an older test run on the same branch.
The test log lists the fifteen slowest tests to guide further improvements.

## Native builds

Run these commands from the repository root:

```bash
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release -DSYNKIT_NATIVE_WHEEL=ON
cmake --build build/native --config Release --parallel 2
cmake --install build/native --config Release --prefix build/native-install
```

The CMake project builds the atom mapper, CRN analysis, graph canonicalization,
and partial-rule application libraries. It requires CMake 3.20 or later, a C++17
compiler, and Boost headers for exact CRN analysis. Set `BOOST_INCLUDE_DIR` if
Boost is outside the compiler's search path.

Disable individual components with `SYNKIT_BUILD_MAPPER`, `SYNKIT_BUILD_CRN`,
`SYNKIT_BUILD_CANON`, or `SYNKIT_BUILD_PARTIAL` set to `OFF`.

## Packages

```bash
python -m build
python scripts/check_package_artifacts.py dist --native-smoke
```

`check_package_artifacts.py` checks wheel and source-archive contents, then
installs the wheel in a temporary environment. `--native-smoke` also exercises
the installed native backends. `--archives-only` skips installation.
`smoke_native_wheel.py` provides the runtime checks used by release workflows.

## Release channels

A GitHub release triggers the PyPI, Conda, and Docker workflows. Its tag, such as
`v1.7.0`, must match `pyproject.toml` and `recipe/meta.yaml`.

- PyPI: platform wheels for Linux, macOS, and Windows.
- Conda: native packages published to the `tieulongphan` channel.
- Docker: Linux AMD64 and ARM64 images.

Manual workflow runs build and validate artifacts without publishing.
