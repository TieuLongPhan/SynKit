"""Build the one final paper package from the integrated SynKit sources."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv, log, cwd=None):
    try:
        with log.open("w") as stream:
            subprocess.run(argv, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as error:
        rescue = Path(tempfile.mkdtemp(prefix="synkit-stereo-failed-build-"))
        shutil.copytree(log.parent, rescue / "build")
        raise RuntimeError("Build failed; diagnostic files retained in " + str(rescue)) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    paper = args.paper_dir.resolve()
    root = paper.parents[1]
    sys.path.insert(0, str(root))
    from Experiment.Stereo.manuscript_tables import build_ledger, write_ledger

    ledger = build_ledger(json.loads((paper / "results/manifest.json").read_text()), root)
    if len(ledger) != 269:
        raise RuntimeError("The selected claim inventory changed.")
    write_ledger(ledger, paper / "results")
    proof = json.loads((paper / "results/proof_binding.json").read_text())
    pattern = re.compile(r"\\begin\{(definition|lemma|theorem|corollary|proposition)\}.*?\\end\{\1\}"
                         r"(?:\s*\\begin\{proof\}.*?\\end\{proof\})?", re.S)
    checked = 0
    for name in proof["source_documents"]:
        blocks = [m.group(0) for m in pattern.finditer((root / name).read_text())]
        for label, entry in proof["labelled_statements_and_immediate_proofs"].items():
            if entry["file"] != name:
                continue
            block = next((b for b in blocks if "\\label{" + label + "}" in b), None)
            if block is None or hashlib.sha256(block.encode()).hexdigest() != entry["statement_and_proof_sha256"]:
                raise RuntimeError("A reviewed mathematical block changed: " + label)
            checked += 1
    if checked != 33:
        raise RuntimeError("The mathematical block inventory changed.")

    tools = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="synkit-stereo-submission-") as temporary:
        work = Path(temporary)
        for driver, destination in [("build_journal_submission.py", "documents"),
                                    ("build_journal_figures.py", "graphics")]:
            run([sys.executable, str(tools / driver), "--paper-dir", str(paper),
                 "--output-dir", str(work / destination)], work / (destination + ".log"))
        documents = work / "documents"
        graphics = work / "graphics"
        doc_manifest = json.loads((documents / "build_manifest.json").read_text())
        fig_manifest = json.loads((graphics / "figure_upload_manifest.json").read_text())
        final = work / "final"
        final.mkdir()
        for name in ["manuscript.pdf", "Additional_file_1.pdf"]:
            shutil.copy2(documents / name, final / name)
        figures = final / "figures"
        figures.mkdir()
        for row in fig_manifest["figures"]:
            shutil.copy2(graphics / row["file"], figures / row["file"])

        # A single editable ZIP covers both documents, all figures and the letter.
        sources = {}
        for archive in [documents / "journal_latex_sources.zip", graphics / "figure_sources.zip"]:
            with zipfile.ZipFile(archive) as packed:
                for name in packed.namelist():
                    data = packed.read(name)
                    if name in sources and sources[name] != data:
                        raise RuntimeError("Conflicting shared source: " + name)
                    sources[name] = data
        sources["cover_letter.tex"] = (paper / "cover_letter.tex").read_bytes()
        sources["figure_captions.tex"] = (graphics / "figure_captions.tex").read_bytes()
        source_inventory = {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                            for name, data in sorted(sources.items())}
        with zipfile.ZipFile(final / "sources.zip", "w", compression=zipfile.ZIP_DEFLATED) as packed:
            for name, data in sorted(sources.items()):
                packed.writestr(name, data)
        letter = work / "letter"
        letter.mkdir()
        for name, data in sources.items():
            target = letter / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "cover_letter.tex"],
            work / "letter.log", cwd=letter)
        shutil.copy2(letter / "cover_letter.pdf", final / "cover_letter.pdf")
        artifacts = {}
        for path in sorted(final.rglob("*")):
            if path.is_file():
                artifacts[str(path.relative_to(final))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        report = {
            "schema": "synkit.stereo-final-submission/1",
            "authoring_source_sha256": sha256(paper / "main.tex"),
            "selected_manifest_sha256": sha256(paper / "results/manifest.json"),
            "numerical_claims": len(ledger), "reviewed_mathematical_blocks": checked,
            "manuscript_pages": doc_manifest["artifacts"]["manuscript.pdf"]["pages"],
            "supporting_information_pages": doc_manifest["artifacts"]["supporting_information.pdf"]["pages"],
            "artifacts": artifacts, "source_inventory": source_inventory,
            "journal_source_inventory": doc_manifest["source_inventory"],
            "figure_source_inventory": fig_manifest["source_inventory"],
            "figures": fig_manifest["figures"],
            "submission_performed": False,
            "boundary": "One local final package. Competing interests remain pending; author approval, public DOI "
                        "and external journal submission are not asserted. Local TeX Live 2023 is tested; the "
                        "publisher's TeX Live 2021 server is untested.",
        }
        (final / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        run([sys.executable, str(tools / "verify_submission.py"), "--package-dir", str(final),
             "--output", str(final / "verification.json")], work / "verification.log")
        output = paper / "submission"
        if output.exists():
            shutil.rmtree(output)
        shutil.copytree(final, output)
        # Convenient root PDFs point to the one verified package, so rebuilding
        # never leaves an older combined manuscript beside the editable source.
        for name, target in [("main.pdf", "manuscript.pdf"),
                             ("technical_details.pdf", "Additional_file_1.pdf"),
                             ("cover_letter.pdf", "cover_letter.pdf")]:
            alias = paper / name
            if alias.exists() or alias.is_symlink():
                alias.unlink()
            alias.symlink_to(Path("submission") / target)
        print(json.dumps({"package": str(output), "figures": len(fig_manifest["figures"]),
                          "editable_source_files": len(sources), "claims": len(ledger), "verified": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
