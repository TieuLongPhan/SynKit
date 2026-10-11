#!/usr/bin/env python3
"""Draw selected ACS identity examples in the paper's thesis-derived style.

The historical four-record composite remains the default.  Use
``--record-ids VS170 VS300 --panels-only`` for the current main-text panels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
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
STYLE_TOKENS = OUTPUT_STEM.parent / "thesis_tokens.tex"


def _style_colors() -> dict[str, tuple[float, float, float]]:
    """Read the self-contained paper copy of the thesis color vocabulary."""
    tokens = dict(re.findall(
        r"\\definecolor\{([^}]+)\}\{HTML\}\{([0-9A-Fa-f]{6})\}",
        STYLE_TOKENS.read_text(),
    ))
    return {
        name: tuple(int(value[index:index + 2], 16) / 255 for index in (0, 2, 4))
        for name, value in tokens.items()
    }


def _configure_drawer(drawer: rdMolDraw2D.MolDraw2DSVG) -> None:
    options = drawer.drawOptions()
    palette = _style_colors()
    options.useBWAtomPalette()
    options.setAtomPalette({
        1: palette["ThesisMuted"],
        5: palette["ChemBoron"],
        6: palette["ThesisText"],
        7: palette["ChemNitrogen"],
        8: palette["ChemOxygen"],
        9: palette["ChemFluorine"],
        14: palette["ChemSilicon"],
        15: palette["ChemPhosphorus"],
        16: palette["ChemSulfur"],
        17: palette["ChemChlorine"],
        35: palette["ChemBromine"],
        53: palette["ChemIodine"],
    })
    options.atomHighlightsAreCircles = True
    options.continuousHighlight = False
    options.fillHighlights = True
    options.flagCloseContactsDist = -1
    options.highlightBondWidthMultiplier = 12
    options.legendFontSize = 25
    options.baseFontSize = 0.72
    options.bondLineWidth = 2.2
    options.padding = 0.08
    options.singleColourWedgeBonds = True


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-ids", nargs="+", choices=REVIEW_IDS,
                        default=REVIEW_IDS)
    parser.add_argument("--panels-only", action="store_true",
                        help="Write individual editable SVG/PDF panels without the historical composite.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_STEM.parent)
    args = parser.parse_args()
    record_ids = tuple(dict.fromkeys(args.record_ids))
    output_stem = args.output_dir.resolve() / OUTPUT_STEM.name
    # A pale carrier halo preserves bond and wedge visibility at print size.
    vermilion = _style_colors()["ThesisVermilion"]
    highlight = tuple(0.18 * channel + 0.82 for channel in vermilion)
    rows = {row["ID"]: row for row in load_dataset()}
    molecules = []
    highlights = []
    colors = []
    radii = []
    legends = []
    checks = []
    for record_id in record_ids:
        row = rows[record_id]
        molecule = _parse_supplied_configured_smiles(row["Input SMILES"])
        if molecule is None:
            raise ValueError(f"Cannot parse {record_id}")
        input_tags = [(atom.GetIdx(), str(atom.GetChiralTag()))
                      for atom in molecule.GetAtoms()
                      if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED]
        input_cip = Chem.FindMolChiralCenters(
            molecule, includeUnassigned=True, useLegacyImplementation=False,
        )
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

        checks.append({"record_id": record_id,
                       "source_smiles": row["Input SMILES"],
                       "reference": manual, "exact_chemical": exact})

        if record_id == "VS300":
            # Make the central [C@H] hydrogen explicit in the artwork so no
            # cage bond can be mistaken for a fifth substituent.
            molecule = Chem.AddHs(molecule, onlyOnAtoms=[1])
            rdCoordGen.AddCoords(molecule)
        else:
            rdDepictor.Compute2DCoords(molecule)
        drawn_tags = [(atom.GetIdx(), str(atom.GetChiralTag()))
                      for atom in molecule.GetAtoms()
                      if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED]
        drawn_cip = Chem.FindMolChiralCenters(
            molecule, includeUnassigned=True, useLegacyImplementation=False,
        )
        drawn_exact = classify_rdkit_stereograph_mirror(
            molecule, require_complete=False,
        ).status.value
        if input_tags != drawn_tags or input_cip != drawn_cip or drawn_exact != exact:
            raise ValueError(f"Drawing preparation altered the configuration of {record_id}")
        configured = [
            atom.GetIdx()
            for atom in molecule.GetAtoms()
            if atom.GetChiralTag()
            in {
                Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
            }
        ]
        checks[-1].update({"highlighted_atoms": configured,
                           "source_and_drawn_chiral_tags_equal": True,
                           "source_and_drawn_cip_equal": True,
                           "source_and_drawn_mirror_equal": True,
                           "drawn_cip": drawn_cip})
        molecules.append(molecule)
        highlights.append(configured)
        colors.append({index: highlight for index in configured})
        radii.append({index: 0.25 for index in configured})
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
            legends.append(f"{record_id}    reference: {manual} | certificate: {exact}")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    if not args.panels_only:
        drawer = rdMolDraw2D.MolDraw2DSVG(1800, 1200, 900, 600)
        _configure_drawer(drawer)
        drawer.DrawMolecules(
            molecules,
            highlightAtoms=highlights,
            highlightAtomColors=colors,
            highlightAtomRadii=radii,
            legends=legends,
        )
        _write_drawing(drawer, output_stem, png_width=1800)

    for record_id, molecule, atoms, atom_colors, atom_radii in zip(
        record_ids,
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
            output_stem.with_name(f"exact_mirror_{record_id.lower()}"),
        )
    print(json.dumps({"record_checks": checks,
                      "panels_only": args.panels_only,
                      "output_directory": str(output_stem.parent),
                      "palette_source": str(STYLE_TOKENS)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
