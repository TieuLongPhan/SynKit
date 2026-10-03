"""Rebuild and compare the figures using only their editable source ZIP."""

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
    report = json.loads((package / "figure_upload_manifest.json").read_text())
    archive = package / "figure_sources.zip"
    require(sha256(archive) == report["source_zip_sha256"], "Figure source ZIP hash differs.")
    with zipfile.ZipFile(archive) as packed:
        require(packed.testzip() is None and set(packed.namelist()) == set(report["source_inventory"]),
                "Figure source ZIP inventory differs.")
        for name, entry in report["source_inventory"].items():
            require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe archive path.")
            require(hashlib.sha256(packed.read(name)).hexdigest() == entry["sha256"],
                    "Figure source byte mismatch: " + name)
        packed.extractall(target)
    rows = []
    for figure in report["figures"]:
        original = package / figure["file"]
        require(sha256(original) == figure["sha256"], "Selected figure bytes differ.")
        stem = original.stem
        with (target / (stem + "_reader_build.log")).open("w") as stream:
            subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                            "-recorder", stem + ".tex"], cwd=target, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        log = (target / (stem + ".log")).read_text()
        require("Overfull" not in log and not re.search(r"(?:Reference|Citation).*undefined", log),
                "Unresolved content or overflow: " + stem)
        cropped = target / (stem + "_cropped.pdf")
        with (target / (stem + "_reader_crop.log")).open("w") as stream:
            subprocess.run(["pdfcrop", "--margins", "1", str(target / (stem + ".pdf")),
                            str(cropped)], stdout=stream, stderr=subprocess.STDOUT, check=True)
        info = subprocess.check_output(["pdfinfo", str(cropped)], text=True)
        require(re.search(r"^Pages:\s+1$", info, re.M) is not None,
                "Reader figure is not a single page: " + stem)
        page_size = re.search(r"^Page size:\s+(.+)$", info, re.M).group(1)
        require(page_size == figure["page_size"], "Reader figure dimensions differ: " + stem)
        fonts = subprocess.check_output(["pdffonts", str(cropped)], text=True).splitlines()[2:]
        require(fonts and all(row.split()[-5] == "yes" for row in fonts if row.strip()),
                "Reader figure font is not embedded: " + stem)
        for source_pdf, suffix in ((original, "selected"), (cropped, "rebuilt")):
            subprocess.run(["pdftoppm", "-png", "-singlefile", "-r", "180", str(source_pdf),
                            str(target / (stem + "_" + suffix))], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        selected_raster = target / (stem + "_selected.png")
        rebuilt_raster = target / (stem + "_rebuilt.png")
        require(sha256(selected_raster) == sha256(rebuilt_raster),
                "Reader figure rendering differs: " + stem)
        percentage = figure.get("rendered_acs_percentage_checked")
        if percentage:
            require(percentage in subprocess.check_output(["pdftotext", str(cropped), "-"], text=True),
                    "Reader ACS percentage differs.")
        rows.append({"figure_number": figure["figure_number"], "file": original.name,
                     "selected_sha256": sha256(original), "rebuilt_sha256": sha256(cropped),
                     "rendered_png_sha256": sha256(rebuilt_raster),
                     "rendering_identical_at_180_dpi": True, "page_size": page_size,
                     "all_fonts_embedded": True, "rendered_acs_percentage_checked": percentage})
    checked = {"schema": "synkit.stereo-journal-figure-source-reader/1", "passed": True,
               "source_zip_sha256": sha256(archive), "source_files_checked": len(report["source_inventory"]),
               "reader_driver_sha256": sha256(Path(__file__)), "figures": rows,
               "tex_version": subprocess.check_output(["pdflatex", "--version"], text=True).splitlines()[0],
               "boundary": (
                  "All cropped vector figures rebuild from their checked editable "
                  "archive with identical 180 dpi renderings. This verifies local source "
                  "portability, not independent human review or the journal-server compiler.")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(checked, indent=2) + "\n")
    print(json.dumps({"passed": True, "source_files": checked["source_files_checked"],
                      "identical_figure_renderings": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
