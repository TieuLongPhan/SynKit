# Repository scripts

Run these entry points from any working directory with the project environment
activated. They use `python`, or the interpreter selected by `PYTHON`.

- `lint.sh`: source-size, Sphinx docstring style, and Flake8 checks.
- `pytest.sh`: all library tests in `Test/`, or supplied pytest arguments.
- `build_doc.sh`: strict Sphinx HTML build into `docs/`.
- `check_package_artifacts.py`: inspect the wheel and source archive, then
  verify the installed wheel; `--mapper-smoke` also checks Python/C++ mapping.

`check_python_file_size.py`, `python_file_size_baseline.json`, and
`check_docstring_style.py` support the lint command.

```bash
conda activate synkit
bash scripts/lint.sh
bash scripts/pytest.sh
```

Mapping campaigns, dataset selection, benchmarks, and figure programs moved to
the sibling repository at `../Synister/synkit/Experiment/Synister/`.
