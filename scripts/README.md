# Repository scripts

Run these entry points from any working directory with the project environment
activated. They use `python`, or the interpreter selected by `PYTHON`.

- `lint.sh`: source-size, Sphinx docstring style, and Flake8 checks.
- `pytest.sh`: all library tests in `Test/`, or supplied pytest arguments.
- `build_doc.sh`: strict Sphinx HTML build into `docs/`.
- `check_package_artifacts.py`: inspect the wheel and source archive, then
  verify the installed wheel. `--native-smoke` checks mapping, CRN analysis,
  and graph/stereograph canonicalization without a compiler.
- `smoke_native_wheel.py`: the installed-package check shared by PyPI, Conda,
  and Docker validation.

## Native builds

The top-level CMake project builds the three production libraries together:
the atom mapper (including its solvers), exact CRN analysis, and graph
canonicalization (also used by stereographs).

```bash
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release -DSYNKIT_NATIVE_WHEEL=ON
cmake --build build/native --config Release --parallel 2
cmake --install build/native --config Release --prefix build/native-install
```

The build needs CMake 3.20 or later, a C++17 compiler, and Boost headers for CRN analysis.
Set `BOOST_INCLUDE_DIR` to the directory containing `boost/` if needed.
`SYNKIT_BUILD_MAPPER`, `SYNKIT_BUILD_CRN`, and `SYNKIT_BUILD_CANON` can each be
set to `OFF` for a focused developer build. Existing component builders remain
available for that purpose.

`python -m build` uses the same top-level CMake project once per wheel. It
packages all three libraries and preserves their separate Python APIs and
optional SDK settings. Platform-wheel users need no compiler, CMake, or Boost
installation to run these backends. The release workflow repairs and checks
the installed wheels on Linux, macOS, and Windows.

```bash
python -m build
python scripts/check_package_artifacts.py dist --native-smoke
```

`--mapper-smoke` remains an alias for `--native-smoke`.

## Release channels

Publishing a GitHub release triggers the PyPI, Conda, and Docker workflows.
The release tag must match `pyproject.toml`, for example `v1.6.5`. Each workflow
builds the three C++ backends through the shared CMake project and tests the
installed package before publishing its artifacts.

- PyPI builds platform wheels for Linux, macOS, and Windows and checks them
  with Python 3.10–3.13.
- Conda builds a native package for each supported platform and Python minor
  version. Dependencies come from conda-forge; packages are published to the
  `tieulongphan` channel. Conda and Mamba may rewrite library paths during
  installation, so native integrity checks use their final installed-file
  hashes in `conda-meta`.
- Docker builds native Linux AMD64 and ARM64 images. The runtime image contains
  the compiled libraries without compilers, CMake, or Boost headers. The
  release workflow publishes the exact tested images and combines their
  platform manifests.

Manual workflow runs validate the artifacts without publishing. The Conda
workflow checks `ANACONDA_TOKEN`, and Docker checks its Docker Hub login, when
those credentials are available. Release publication
uses that secret for Anaconda.org, `DOCKERHUB_USER` and `DOCKERHUB_TOKEN` for
Docker Hub, and the existing `pypi` trusted-publishing environment for PyPI.

`check_python_file_size.py`, `python_file_size_baseline.json`, and
`check_docstring_style.py` support the lint command.

```bash
conda activate synkit
bash scripts/lint.sh
bash scripts/pytest.sh
```

Mapping campaigns, dataset selection, benchmarks, and figure programs moved to
the sibling repository at `../Synister/synkit/Experiment/Synister/`.
