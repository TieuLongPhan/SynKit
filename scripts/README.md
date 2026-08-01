# Repository scripts

Run these entrypoints from any working directory:

- `lint.sh`: Python file-size policy and Flake8 checks;
- `pytest.sh`: the full test suite, or supplied pytest arguments;
- `build_doc.sh`: strict Sphinx HTML documentation build.

`check_python_file_size.py` and `python_file_size_baseline.json` support the
lint entrypoint and its regression tests.
