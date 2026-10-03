"""Compile both journal documents using only the checked source ZIP."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package = args.package_dir.resolve()
    target = args.extraction_dir.resolve()
    require(not target.exists(), "Use a new extraction directory.")
    report = json.loads((package / "build_manifest.json").read_text())
    archive = package / "journal_latex_sources.zip"
    require(sha256(archive) == report["artifacts"][archive.name]["sha256"], "Source ZIP hash differs.")
    with zipfile.ZipFile(archive) as packed:
        require(packed.testzip() is None and set(packed.namelist()) == set(report["source_inventory"]),
                "Source ZIP inventory differs.")
        for name, entry in report["source_inventory"].items():
            require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe archive path.")
            require(hashlib.sha256(packed.read(name)).hexdigest() == entry["sha256"], "Source byte mismatch.")
        packed.extractall(target)
    for name in ("manuscript", "supporting_information", "manuscript", "supporting_information"):
        with (target / (name + "_reader_build.log")).open("w") as stream:
            subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                            "-recorder", name + ".tex"], cwd=target, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
    documents = {}
    for name in ("manuscript", "supporting_information"):
        log = (target / (name + ".log")).read_text()
        require(not re.search(r"(?:Reference|Citation).*undefined|There were undefined references", log)
                and "Overfull" not in log, "Unresolved references or overflow in " + name)
        pdf = target / (name + ".pdf")
        info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
        fonts = subprocess.check_output(["pdffonts", str(pdf)], text=True)
        font_rows = fonts.splitlines()[2:]
        require(font_rows and all(line.split()[-5] == "yes" for line in font_rows if line.strip()),
                "A PDF font is not embedded: " + name)
        pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))
        require(pages == report["artifacts"][pdf.name]["pages"], "Reader pagination differs: " + name)
        documents[pdf.name] = {"pages": pages, "sha256": sha256(pdf), "bytes": pdf.stat().st_size,
                               "unresolved_references_or_citations": False, "overfull_boxes": False,
                               "all_fonts_embedded": True}
    checked = {"schema": "synkit.stereo-journal-source-reader/1", "passed": True,
               "source_zip_sha256": sha256(archive), "source_files_checked": len(report["source_inventory"]),
               "reader_driver_sha256": sha256(Path(__file__)), "documents": documents,
               "tex_version": subprocess.check_output(["pdflatex", "--version"], text=True).splitlines()[0],
               "boundary": ("Extracted editable sources compile locally with their supplied dependencies. "
                            "No journal-server TeX Live 2021 compile or independent human review is claimed.")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(checked, indent=2) + "\n")
    print(json.dumps({"passed": True, "source_files": checked["source_files_checked"], "documents": documents}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
