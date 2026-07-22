"""Tests for the repository Python file-size policy."""

from pathlib import Path

from tools.check_python_file_size import check_files, count_non_docstring_lines


def test_count_excludes_only_recognized_docstrings(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text(
        "\n".join(
            [
                '"""module',
                'documentation"""',
                'VALUE = "ordinary string"',
                "",
                "class Example:",
                '    """class documentation"""',
                '    marker = "kept"',
                "",
                "    def method(self):",
                '        """method',
                '        documentation"""',
                "        return 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert count_non_docstring_lines(source) == 7


def test_new_oversized_file_fails(tmp_path: Path) -> None:
    source = tmp_path / "large.py"
    source.write_text("value = 1\n" * 4, encoding="utf-8")

    violations, notices = check_files(
        [source],
        root=tmp_path,
        max_lines=3,
        baseline={},
    )

    assert notices == []
    assert violations == ["large.py: 4 non-docstring lines exceeds limit 3"]


def test_legacy_file_may_shrink_but_not_grow(tmp_path: Path) -> None:
    source = tmp_path / "legacy.py"
    source.write_text("value = 1\n" * 4, encoding="utf-8")

    violations, notices = check_files(
        [source],
        root=tmp_path,
        max_lines=3,
        baseline={"legacy.py": 5},
    )
    assert violations == []
    assert notices == ["legacy.py: 4/5 legacy lines (target 3)"]

    source.write_text("value = 1\n" * 6, encoding="utf-8")
    violations, _ = check_files(
        [source],
        root=tmp_path,
        max_lines=3,
        baseline={"legacy.py": 5},
    )
    assert violations == ["legacy.py: 6 lines exceeds legacy baseline 5 (target 3)"]


def test_baseline_entry_must_be_removed_after_compliance(tmp_path: Path) -> None:
    source = tmp_path / "small.py"
    source.write_text("value = 1\n", encoding="utf-8")

    violations, notices = check_files(
        [source],
        root=tmp_path,
        max_lines=3,
        baseline={"small.py": 5},
    )

    assert notices == []
    assert violations == ["small.py: now has 1 lines; remove its stale baseline entry"]
