"""Export all main-text graphics and retain their editable dependencies."""

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


def caption(block):
    start = block.index("\\caption{") + len("\\caption{")
    level, end = 1, start
    while level and end < len(block):
        if block[end] in "{}" and block[end - 1] != "\\":
            level += 1 if block[end] == "{" else -1
        end += 1
    require(level == 0, "Unclosed figure caption.")
    return block[start:end - 1], start - len("\\caption{"), end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paper = args.paper_dir.resolve()
    output = args.output_dir.resolve()
    require(not output.exists(), "Use a new figure output directory.")
    snapshot = output / "source"
    snapshot.mkdir(parents=True)
    previews = output / "previews"
    previews.mkdir()
    source = (paper / "main.tex").read_text()
    preamble, narrative = source.split("\\begin{document}", 1)
    narrative = narrative.split("\\appendix", 1)[0]
    figures = list(re.finditer(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", narrative, re.S))
    require(len(figures) == 11, "The current main figure inventory changed.")
    require("\\documentclass[pdflatex,sn-mathphys-num,referee,lineno]{sn-jnl}" in preamble,
            "Expected the checked article class options.")
    # Keep the manuscript class and its font sizes. Only the available graphic
    # width and page furniture change before the final whitespace crop.
    figure_preamble = preamble.replace("[pdflatex,sn-mathphys-num,referee,lineno]",
                                       "[pdflatex,sn-mathphys-num]")
    figure_preamble = re.sub(r"^\\(?:usepackage\{xr-hyper\}|externaldocument\{[^}]+\})\s*$",
                             "", figure_preamble, flags=re.M)
    figure_preamble += "\n\\pagestyle{empty}\n\\setlength{\\textwidth}{170mm}\n"
    shutil.copy2(paper / "sn-jnl.cls", snapshot / "sn-jnl.cls")
    # Retain local inputs; the final source ZIP is restricted to dependencies
    # actually observed by each figure's TeX recorder.
    for base in (paper / "fig",):
        for path in base.glob("*"):
            if path.is_file() and path.suffix in {".tex", ".pdf"}:
                target = snapshot / path.relative_to(paper)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    values = snapshot / "results/numerical_values.tex"
    values.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paper / "results/numerical_values.tex", values)
    selected_values = {row["claim_id"]: row["value"] for row in
                       json.loads((paper / "results/numerical_ledger.json").read_text())}
    rows, inputs, mentions = [], set(), []
    caption_lines = ["% Figure titles and legends are retained in the main manuscript."]
    for number, match in enumerate(figures, 1):
        block = match.group(0)
        text, start, end = caption(block)
        labels = re.findall(r"\\label\{([^}]+)\}", block)
        require(len(labels) == 1, "Expected exactly one outer figure label.")
        label = labels[0]
        first_mention = narrative.find("\\ref{" + label + "}")
        require(first_mention >= 0 and first_mention < match.start(),
                "Figure is not introduced before its placement: " + label)
        mentions.append(first_mention)
        title, legend = text.split(".", 1)
        title = " ".join(title.split()) + "."
        legend = " ".join(legend.split())
        # Titles contain plain words. For legends, count both source tokens and
        # a conservative lexical form that also counts TeX command names.
        title_words = len(title.split())
        legend_words = max(len(legend.split()), len(re.findall(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*", legend)))
        require(title_words <= 15 and legend_words <= 300, "Figure title or legend exceeds journal limit: " + label)
        body = block[:start] + block[end:]
        body = re.sub(r"\\(?:begin|end)\{figure\*?\}(?:\[[^]]*\])?", "", body)
        body = re.sub(r"\\label\{[^}]+\}", "", body)
        stem = "Figure_" + str(number).zfill(2)
        tex = snapshot / (stem + ".tex")
        tex.write_text(figure_preamble + "\n\\begin{document}\n\\noindent\\begin{minipage}{170mm}\n"
                       + body + "\n\\end{minipage}\n\\end{document}\n")
        with (snapshot / (stem + "_build.log")).open("w") as stream:
            subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                            "-recorder", tex.name], cwd=snapshot, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        log = (snapshot / (stem + ".log")).read_text()
        require("Overfull" not in log and not re.search(r"(?:Reference|Citation).*undefined", log),
                "Figure source has unresolved content or overflow: " + label)
        raw = snapshot / (stem + ".pdf")
        pdf = output / (stem + ".pdf")
        with (snapshot / (stem + "_crop.log")).open("w") as stream:
            subprocess.run(["pdfcrop", "--margins", "1", str(raw), str(pdf)], stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
        require(re.search(r"^Pages:\s+1$", info, re.M) is not None and pdf.stat().st_size <= 10_000_000,
                "Figure page count or file size exceeds the upload contract.")
        fonts = subprocess.check_output(["pdffonts", str(pdf)], text=True).splitlines()[2:]
        require(fonts and all(row.split()[-5] == "yes" for row in fonts if row.strip()),
                "A figure font is not embedded.")
        rendered_percentage = None
        if label == "fig:benchmark-evidence-map":
            rendered_percentage = format(100 * selected_values["AcsSourceCorrect"] /
                                         selected_values["AcsSourceRecords"], ".2f") + "%"
            graphic_text = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True)
            require(rendered_percentage in graphic_text, "Rendered ACS agreement percentage differs from the ledger.")
        subprocess.run(["pdftoppm", "-png", "-singlefile", "-r", "180", str(pdf),
                        str(previews / stem)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for line in (snapshot / (stem + ".fls")).read_text().splitlines():
            if not line.startswith("INPUT "):
                continue
            path = Path(line[6:])
            path = (path if path.is_absolute() else snapshot / path).resolve()
            if path.is_relative_to(snapshot) and path.is_file() and (
                    path.suffix in {".tex", ".cls", ".sty"} or
                    (path.suffix == ".pdf" and path.parent != snapshot)):
                inputs.add(path.relative_to(snapshot))
        rows.append({"figure_number": number, "label": label, "file": pdf.name,
                     "sha256": sha256(pdf), "bytes": pdf.stat().st_size, "pages": 1,
                     "title_tex": title, "legend_tex": legend, "title_words": title_words,
                     "conservative_legend_words": legend_words, "all_fonts_embedded": True,
                     "caption_removed_from_graphic": True, "crop_margin_points": 1,
                     "rendered_acs_percentage_checked": rendered_percentage,
                     "page_size": re.search(r"^Page size:\s+(.+)$", info, re.M).group(1),
                     "preview": str((previews / (stem + ".png")).relative_to(output))})
        caption_lines.extend(["% " + stem + " / " + label, "\\caption{" + text + "}"])
    require(mentions == sorted(mentions), "Figure upload order differs from first textual mention.")
    (output / "figure_captions.tex").write_text("\n\n".join(caption_lines) + "\n")
    inventory = {str(path): {"sha256": sha256(snapshot / path), "bytes": (snapshot / path).stat().st_size}
                 for path in sorted(inputs)}
    archive = output / "figure_sources.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as packed:
        for name in inventory:
            packed.write(snapshot / name, name)
    with zipfile.ZipFile(archive) as packed:
        require(packed.testzip() is None and set(packed.namelist()) == set(inventory),
                "Figure source ZIP inventory differs.")
        for name, entry in inventory.items():
            require(hashlib.sha256(packed.read(name)).hexdigest() == entry["sha256"],
                    "A figure source ZIP byte differs.")
    report = {"schema": "synkit.stereo-journal-figure-uploads/1", "builder_sha256": sha256(Path(__file__)),
              "main_tex_sha256": sha256(paper / "main.tex"), "numerical_values_sha256": sha256(values),
              "figures": rows, "figures_in_first_mention_order": True,
              "source_inventory": inventory, "source_zip_sha256": sha256(archive),
              "all_figure_source_bytes_checked": True, "vector_pdf_graphics": True,
              "visual_review_completed": False,
              "boundary": (
                  "Graphics are compiled from the current source and retained measurements. "
                  "Cropping only removes surrounding whitespace. Panel keys remain; titles "
                  "and legends are in the manuscript. Human author review remains separate.")}
    (output / "figure_upload_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"figures": len(rows), "editable_source_files": len(inventory),
                      "all_fonts_embedded": True, "all_size_and_caption_limits_passed": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
