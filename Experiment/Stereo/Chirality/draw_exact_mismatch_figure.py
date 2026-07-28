#!/usr/bin/env python3
"""Draw three corrected identity cases and the remaining ACS difference."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from rdkit import Chem
from rdkit.Chem import rdCoordGen, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Stereo.Chirality.exact_mirror import (  # noqa: E402
    _parse_supplied_configured_smiles,
)
from Experiment.Stereo.Chirality.published import load_dataset  # noqa: E402
from synkit.Graph.Stereo import (  # noqa: E402
    classify_rdkit_stereograph_mirror,
)
from synkit.Chem.Molecule.global_stereo import (  # noqa: E402
    analyze_global_stereo_support,
)

REVIEW_IDS = ("VS170", "VS215", "VS216", "VS300")
EXPECTED_CHEMICAL = {
    "VS170": "chiral",
    "VS215": "achiral",
    "VS216": "achiral",
    "VS300": "achiral",
}
OUTPUT_STEM = ROOT / "paper" / "stereo_graph" / "fig" / "exact_mirror_mismatches"
HIGHLIGHT = (0.835, 0.369, 0.0)


def _configure_drawer(drawer: rdMolDraw2D.MolDraw2DSVG) -> None:
    options = drawer.drawOptions()
    options.atomHighlightsAreCircles = True
    options.continuousHighlight = False
    options.fillHighlights = True
    options.flagCloseContactsDist = -1
    options.highlightBondWidthMultiplier = 12
    options.legendFontSize = 25
    options.baseFontSize = 0.72
    options.bondLineWidth = 2.2
    options.padding = 0.08


def _write_drawing(
    drawer: rdMolDraw2D.MolDraw2DSVG,
    stem: Path,
    *,
    png_width: int | None = None,
) -> None:
    drawer.FinishDrawing()
    svg_path = stem.with_suffix(".svg")
    pdf_path = stem.with_suffix(".pdf")
    raw_pdf_path = stem.with_name(f"{stem.name}_raw.pdf")
    svg_path.write_text(drawer.GetDrawingText(), encoding="utf-8")
    subprocess.run(
        (
            "rsvg-convert",
            "-b",
            "white",
            "-f",
            "pdf",
            "-o",
            str(raw_pdf_path),
            str(svg_path),
        ),
        check=True,
    )
    subprocess.run(
        (
            "gs",
            "-q",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.5",
            f"-sOutputFile={pdf_path}",
            str(raw_pdf_path),
        ),
        check=True,
    )
    raw_pdf_path.unlink()
    if png_width is not None:
        subprocess.run(
            (
                "rsvg-convert",
                "-b",
                "white",
                "-f",
                "png",
                "-w",
                str(png_width),
                "-o",
                str(stem.with_suffix(".png")),
                str(svg_path),
            ),
            check=True,
        )


def main() -> int:
    rows = {row["ID"]: row for row in load_dataset()}
    molecules = []
    highlights = []
    colors = []
    radii = []
    legends = []
    for record_id in REVIEW_IDS:
        row = rows[record_id]
        molecule = _parse_supplied_configured_smiles(row["Input SMILES"])
        if molecule is None:
            raise ValueError(f"Cannot parse {record_id}")
        result = classify_rdkit_stereograph_mirror(
            molecule,
            require_complete=False,
        )
        manual = row["manual"].lower()
        exact = result.status.value
        if exact != EXPECTED_CHEMICAL[record_id]:
            raise ValueError(
                f"{record_id} returned {exact}, expected "
                f"{EXPECTED_CHEMICAL[record_id]}"
            )

        if record_id == "VS300":
            # Make the central [C@H] hydrogen explicit in the artwork so no
            # cage bond can be mistaken for a fifth substituent.
            molecule = Chem.AddHs(molecule, onlyOnAtoms=[1])
            rdCoordGen.AddCoords(molecule)
        else:
            rdDepictor.Compute2DCoords(molecule)
        configured = [
            atom.GetIdx()
            for atom in molecule.GetAtoms()
            if atom.GetChiralTag()
            in {
                Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
            }
        ]
        molecules.append(molecule)
        highlights.append(configured)
        colors.append({index: HIGHLIGHT for index in configured})
        radii.append({index: 0.32 for index in configured})
        if record_id == "VS300":
            sanitized = Chem.MolFromSmiles(row["Input SMILES"])
            if sanitized is None:
                raise ValueError("Cannot parse sanitized VS300")
            global_certificate = analyze_global_stereo_support(sanitized)
            global_text = (
                "chiral, orientation unspecified"
                if global_certificate.necessarily_chiral
                else global_certificate.state.value
            )
            legends.append(
                f"{record_id}    reference: {manual} | source: {exact} | "
                f"global: {global_text}"
            )
        else:
            legends.append(
                f"{record_id}    reference: {manual} | certificate: {exact}"
            )

    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    drawer = rdMolDraw2D.MolDraw2DSVG(1800, 1200, 900, 600)
    _configure_drawer(drawer)
    drawer.DrawMolecules(
        molecules,
        highlightAtoms=highlights,
        highlightAtomColors=colors,
        highlightAtomRadii=radii,
        legends=legends,
    )
    _write_drawing(drawer, OUTPUT_STEM, png_width=1800)

    for record_id, molecule, atoms, atom_colors, atom_radii in zip(
        REVIEW_IDS,
        molecules,
        highlights,
        colors,
        radii,
    ):
        panel = rdMolDraw2D.MolDraw2DSVG(900, 560)
        _configure_drawer(panel)
        panel.DrawMolecule(
            molecule,
            highlightAtoms=atoms,
            highlightAtomColors=atom_colors,
            highlightAtomRadii=atom_radii,
        )
        _write_drawing(
            panel,
            OUTPUT_STEM.with_name(f"exact_mirror_{record_id.lower()}"),
        )
    print(OUTPUT_STEM.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
