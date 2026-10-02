"""Compile portable vector figures and concatenate a figure-only review PDF."""

from pathlib import Path
import shutil
import subprocess

import fitz

PAPER = Path(__file__).resolve().parents[1] / "paper/synister"
NAMES = (
    "workflow",
    "worked_unmapped",
    "worked_its_classes",
    "exact_pruning",
    "mapping_classes",
    "centre_certainty",
    "context_radius",
    "hydrogen_objective",
    "pilot_spectrum",
    "pilot_structure",
)


def main():
    # Avoid mirroring figures/: TeX searches the output directory before the
    # source tree and would otherwise mistake a wrapper for an included figure.
    build = PAPER / ".build/figure_exports"
    build.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        source = PAPER / "figures" / f"{name}.tex"
        if not source.exists():
            continue
        wrapper = build / f"{name}.tex"
        wrapper.write_text(
            r"\documentclass[border=2pt]{standalone}"
            "\n"
            r"\usepackage{lmodern,amsmath,amssymb,graphicx,tikz}"
            "\n"
            r"\usetikzlibrary{arrows.meta,calc,positioning}"
            "\n"
            r"\input{notation.tex}\input{figures/figure_style.tex}"
            "\n"
            r"\begin{document}\setlength{\linewidth}{6.5in}"
            "\n"
            rf"\resizebox{{\linewidth}}{{!}}{{\input{{figures/{name}.tex}}}}"
            "\n"
            r"\end{document}"
            "\n"
        )
        result = subprocess.run(
            [
                "pdflatex",
                "-interaction=batchmode",
                "-halt-on-error",
                f"-output-directory={build}",
                str(wrapper),
            ],
            cwd=PAPER,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode:
            raise RuntimeError((build / f"{name}.log").read_text()[-5000:])
        shutil.copy2(build / f"{name}.pdf", PAPER / "figures" / f"{name}.pdf")
    with fitz.open() as gallery:
        for name in NAMES:
            with fitz.open(PAPER / "figures" / f"{name}.pdf") as figure:
                gallery.insert_pdf(figure)
        gallery.save(PAPER / "figure_gallery.pdf", garbage=4, deflate=True)
    print(f"Wrote {len(NAMES)} vector figures and paper/synister/figure_gallery.pdf")


if __name__ == "__main__":
    main()
