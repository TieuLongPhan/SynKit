"""Build separate, numbered journal manuscript and supporting-information PDFs.

Scientific text is taken directly from the two standalone authoring documents.
Only build names and external-reference wiring change.
The source ZIP contains every local dependency observed by the TeX recorder.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def build(snapshot, name):
    with (snapshot / (name + "_build.log")).open("w") as stream:
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                        "-recorder", name + ".tex"], cwd=snapshot, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def recorded_inputs(snapshot, name):
    files = set()
    for line in (snapshot / (name + ".fls")).read_text().splitlines():
        if not line.startswith("INPUT "):
            continue
        path = Path(line[6:])
        path = path if path.is_absolute() else snapshot / path
        path = path.resolve()
        if path.is_relative_to(snapshot) and path.is_file():
            files.add(path.relative_to(snapshot))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paper = args.paper_dir.resolve()
    out = args.output_dir.resolve()
    snapshot = out / "source"
    require(not out.exists(), "Use a new output directory to preserve previous builds.")
    snapshot.mkdir(parents=True)
    source = (paper / "main.tex").read_text()
    supporting = (paper / "technical_details.tex").read_text()
    require(re.findall(r"^\\section\{([^}]+)\}", source, re.M) ==
            ["Introduction", "Methods", "Results", "Discussion", "Conclusions"],
            "The journal manuscript section hierarchy changed.")
    for name, text in [("main", source), ("technical_details", supporting)]:
        require(text.count("\\begin{document}") == 1 and text.count("\\end{document}") == 1,
                "Expected a standalone document: " + name)
        require("[pdflatex,sn-mathphys-num,referee,lineno]" in text,
                "Expected double spacing and line numbering: " + name)
        require(text.count("\\bibliography{references}") == 1,
                "Bibliography boundary changed: " + name)
        require("\\clearpage" not in text and "\\newpage" not in text,
                "An explicit author page break remains: " + name)
    require("\\appendix" not in source and supporting.count("\\appendix") == 1,
            "Supporting information must be separate from the main article.")
    require(source.count("\\externaldocument{technical_details}") == 1 and
            supporting.count("\\externaldocument{main}") == 1,
            "Expected reciprocal cross-document references.")
    for path in sorted(paper.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(paper)
        if path.suffix in {".tex", ".bib", ".cls", ".bst", ".sty"} or (
                path.suffix == ".pdf" and (relative.parts[0] == "fig" or
                                           relative.parts[:3] == ("revision", "generated", "performance"))):
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    (snapshot / "manuscript.tex").write_text(
        source.replace("\\externaldocument{technical_details}", "\\externaldocument{supporting_information}"))
    (snapshot / "supporting_information.tex").write_text(
        supporting.replace("\\externaldocument{main}", "\\externaldocument{manuscript}"))
    # Cross-document auxiliary files are established before final compilation.
    for name in ("manuscript", "supporting_information", "manuscript", "supporting_information"):
        build(snapshot, name)
    artifacts = {}
    inputs = set()
    for name in ("manuscript", "supporting_information"):
        log = (snapshot / (name + ".log")).read_text()
        require(not re.search(r"(?:Reference|Citation).*undefined|There were undefined references", log),
                "Unresolved journal references: " + name)
        require("Overfull" not in log, "Overfull journal layout: " + name)
        info = subprocess.check_output(["pdfinfo", str(snapshot / (name + ".pdf"))], text=True)
        pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))
        target = out / (name + ".pdf")
        shutil.copy2(snapshot / target.name, target)
        artifacts[target.name] = {"sha256": sha256(target), "bytes": target.stat().st_size, "pages": pages,
                                  "undefined_references_or_citations": False, "overfull_boxes": False}
        inputs |= recorded_inputs(snapshot, name)
        inputs.update({Path(name + ".aux"), Path(name + ".bbl"), Path(name + ".tex")})
    # The journal asks for an explicitly named additional file. Keep the
    # authoring/build name for cross-document TeX references and provide an
    # identical PDF under its upload name.
    upload = out / "Additional_file_1.pdf"
    shutil.copy2(out / "supporting_information.pdf", upload)
    artifacts[upload.name] = dict(artifacts["supporting_information.pdf"])
    artifacts[upload.name]["identical_to"] = "supporting_information.pdf"
    # BibTeX inputs are not all listed by pdfTeX's recorder.
    inputs.add(Path("references.bib"))
    inputs |= {path.relative_to(snapshot) for path in snapshot.glob("*.bst")}
    inputs = {path for path in inputs if path.suffix not in {".pdf", ".log", ".out", ".fdb_latexmk", ".fls"}
              or path.parts[0] in {"fig", "revision"}}
    source_inventory = {str(path): {"sha256": sha256(snapshot / path), "bytes": (snapshot / path).stat().st_size}
                        for path in sorted(inputs)}
    archive = out / "journal_latex_sources.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as packed:
        for path in sorted(inputs):
            packed.write(snapshot / path, str(path))
    with zipfile.ZipFile(archive) as packed:
        require(packed.testzip() is None and set(packed.namelist()) == set(source_inventory),
                "Source ZIP inventory failed.")
        for name, entry in source_inventory.items():
            require(hashlib.sha256(packed.read(name)).hexdigest() == entry["sha256"],
                    "Source ZIP byte mismatch: " + name)
    artifacts[archive.name] = {"sha256": sha256(archive), "bytes": archive.stat().st_size,
                               "files": len(source_inventory), "all_archived_bytes_verified": True}
    report = {"schema": "synkit.stereo-journal-submission-build/1",
              "authoring_main_sha256": sha256(paper / "main.tex"),
              "authoring_technical_details_sha256": sha256(paper / "technical_details.tex"),
              "builder_sha256": sha256(Path(__file__)), "artifacts": artifacts,
              "source_inventory": source_inventory,
              "class_options": ["pdflatex", "sn-mathphys-num", "referee", "lineno"],
              "explicit_author_page_breaks_removed": True,
              "algorithm_float_placement": "htbp; scientific pseudocode unchanged",
              "main_and_supporting_information_are_standalone_sources": True,
              "main_section_headings": ["Introduction", "Methods", "Results", "Discussion", "Conclusions"],
              "supplemental_statement_equation_figure_table_numbers":
              "S prefix distinguishes cross-document references",
              "journal_additional_file_upload_name": "Additional_file_1.pdf",
              "final_result_selection_and_author_metadata_confirmed": False,
              "claim_boundary": (
                  "A formatted review package; scientific result selection is checked "
                  "separately, while author declarations, approval and the exact release "
                  "DOI remain open.")}
    (out / "build_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"artifacts": artifacts, "output": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
