"""Rebuild the final package independently from its one editable source ZIP."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package = args.package_dir.resolve()
    manifest = json.loads((package / "manifest.json").read_text())
    for name, entry in manifest["artifacts"].items():
        if sha256(package / name) != entry["sha256"]:
            raise RuntimeError("Selected artifact changed: " + name)
    with zipfile.ZipFile(package / "sources.zip") as archive:
        if archive.testzip() or set(archive.namelist()) != set(manifest["source_inventory"]):
            raise RuntimeError("The editable source inventory changed.")
        sources = {}
        for name, entry in manifest["source_inventory"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise RuntimeError("Unsafe source path.")
            sources[name] = archive.read(name)
            if hashlib.sha256(sources[name]).hexdigest() != entry["sha256"]:
                raise RuntimeError("Source bytes changed: " + name)
    tools = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="synkit-stereo-reader-") as temporary:
        work = Path(temporary)
        documents, figures = work / "documents", work / "figures"
        documents.mkdir()
        figures.mkdir()
        for destination, inventory, archive_name in [
            (documents, manifest["journal_source_inventory"], "journal_latex_sources.zip"),
            (figures, manifest["figure_source_inventory"], "figure_sources.zip"),
        ]:
            with zipfile.ZipFile(destination / archive_name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, entry in inventory.items():
                    data = sources[name]
                    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                        raise RuntimeError("Reader subset bytes differ: " + name)
                    archive.writestr(name, data)
        shutil.copy2(package / "manuscript.pdf", documents / "manuscript.pdf")
        shutil.copy2(package / "Additional_file_1.pdf", documents / "supporting_information.pdf")
        source_zip = documents / "journal_latex_sources.zip"
        doc_manifest = {
            "source_inventory": manifest["journal_source_inventory"],
            "artifacts": {
                "journal_latex_sources.zip": {"sha256": sha256(source_zip)},
                "manuscript.pdf": {"pages": manifest["manuscript_pages"]},
                "supporting_information.pdf": {"pages": manifest["supporting_information_pages"]},
            },
        }
        (documents / "build_manifest.json").write_text(json.dumps(doc_manifest))
        for row in manifest["figures"]:
            shutil.copy2(package / "figures" / row["file"], figures / row["file"])
        fig_manifest = {"source_inventory": manifest["figure_source_inventory"],
                        "source_zip_sha256": sha256(figures / "figure_sources.zip"),
                        "figures": manifest["figures"]}
        (figures / "figure_upload_manifest.json").write_text(json.dumps(fig_manifest))
        for driver, location in [("verify_journal_source_archive.py", documents),
                                 ("verify_journal_figure_archive.py", figures)]:
            with (work / (location.name + ".log")).open("w") as log:
                subprocess.run([sys.executable, str(tools / driver), "--package-dir", str(location),
                                "--extraction-dir", str(work / (location.name + "_extracted")),
                                "--output", str(work / (location.name + ".json"))],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        # The cover-letter source shares the same checked dependencies.
        letter = work / "letter"
        letter.mkdir()
        for name, data in sources.items():
            target = letter / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        with (work / "letter.log").open("w") as log:
            subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "cover_letter.tex"],
                           cwd=letter, stdout=log, stderr=subprocess.STDOUT, check=True)

        def text(path):
            return subprocess.check_output(["pdftotext", "-layout", str(path), "-"])

        if text(letter / "cover_letter.pdf") != text(package / "cover_letter.pdf"):
            raise RuntimeError("The letter text differs after source extraction.")
        for name, selected in [("manuscript.pdf", "manuscript.pdf"),
                               ("supporting_information.pdf", "Additional_file_1.pdf")]:
            if text(work / "documents_extracted" / name) != text(package / selected):
                raise RuntimeError("The document text differs after source extraction: " + name)
        report = {
            "schema": "synkit.stereo-final-submission-verification/1", "passed": True,
            "source_zip_sha256": sha256(package / "sources.zip"),
            "source_files_checked": len(sources),
            "documents": json.loads((work / "documents.json").read_text()),
            "figures": json.loads((work / "figures.json").read_text()),
            "document_and_letter_text_identical": True,
            "submission_performed": False,
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"passed": True, "editable_sources": len(sources), "figures": 11}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
