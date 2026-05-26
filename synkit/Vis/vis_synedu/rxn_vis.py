from __future__ import annotations

from dataclasses import dataclass
import html
from typing import Any, Dict, List, Optional, Tuple, Union

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdChemReactions


@dataclass(frozen=True)
class RxnHighlights:
    """Atom/bond highlights keyed by reactant/product molecule index in the reaction."""

    r_atoms: Dict[int, List[int]]
    r_bonds: Dict[int, List[int]]
    p_atoms: Dict[int, List[int]]
    p_bonds: Dict[int, List[int]]


def _ensure_2d(m: Chem.Mol) -> Chem.Mol:
    """Ensure the molecule has 2D coords (in-place)."""
    if m is None:
        return m
    if m.GetNumConformers() == 0:
        # RDKit tends to do better with this than Compute2DCoords alone for some cases
        AllChem.Compute2DCoords(m)
    return m


def _bond_signature(m: Chem.Mol) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    Return mapping: (map_i, map_j) -> (bondTypeInt, isAromaticInt)
    Only for bonds where both atoms have atom-map numbers.
    """
    out: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for b in m.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        m1, m2 = a1.GetAtomMapNum(), a2.GetAtomMapNum()
        if m1 <= 0 or m2 <= 0:
            continue
        key = (m1, m2) if m1 < m2 else (m2, m1)
        # bond type as an int-ish bucket + aromatic flag
        bt = int(b.GetBondTypeAsDouble() * 10)  # e.g., single=10, double=20
        ar = 1 if b.GetIsAromatic() else 0
        out[key] = (bt, ar)
    return out


def _mapnum_to_atomidx(m: Chem.Mol) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for a in m.GetAtoms():
        mn = a.GetAtomMapNum()
        if mn > 0:
            out[mn] = a.GetIdx()
    return out


def _find_changed_bonds_by_atommap(  # noqa: C901
    rxn: rdChemReactions.ChemicalReaction,
) -> Optional[RxnHighlights]:
    """
    Detect changed bonds (formed/broken/changed order) using atom-map numbers.
    If the reaction has no atom maps, return None.
    """
    reactants = [m for m in rxn.GetReactants()]
    products = [m for m in rxn.GetProducts()]

    # If we have essentially no mapping info, bail
    total_mapped = 0
    for m in reactants + products:
        total_mapped += sum(1 for a in m.GetAtoms() if a.GetAtomMapNum() > 0)
    if total_mapped == 0:
        return None

    # Build global bond signatures for each side
    r_sig: Dict[Tuple[int, int], Tuple[int, int]] = {}
    p_sig: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for m in reactants:
        r_sig.update(_bond_signature(m))
    for m in products:
        p_sig.update(_bond_signature(m))

    changed_pairs = set(r_sig.keys()) | set(p_sig.keys())
    changed_pairs = {k for k in changed_pairs if r_sig.get(k) != p_sig.get(k)}

    # Precompute mapnum->atomidx per molecule, and mapnum->(mol_i, atom_i)
    r_mn_loc: Dict[int, Tuple[int, int]] = {}
    p_mn_loc: Dict[int, Tuple[int, int]] = {}
    r_mn2aidx: List[Dict[int, int]] = []
    p_mn2aidx: List[Dict[int, int]] = []

    for i, m in enumerate(reactants):
        d = _mapnum_to_atomidx(m)
        r_mn2aidx.append(d)
        for mn, aidx in d.items():
            r_mn_loc[mn] = (i, aidx)

    for i, m in enumerate(products):
        d = _mapnum_to_atomidx(m)
        p_mn2aidx.append(d)
        for mn, aidx in d.items():
            p_mn_loc[mn] = (i, aidx)

    # Collect atom + bond highlights per molecule index
    r_atoms: Dict[int, List[int]] = {i: [] for i in range(len(reactants))}
    r_bonds: Dict[int, List[int]] = {i: [] for i in range(len(reactants))}
    p_atoms: Dict[int, List[int]] = {i: [] for i in range(len(products))}
    p_bonds: Dict[int, List[int]] = {i: [] for i in range(len(products))}

    def _add_atom(side_atoms: Dict[int, List[int]], mol_i: int, atom_i: int) -> None:
        if atom_i not in side_atoms[mol_i]:
            side_atoms[mol_i].append(atom_i)

    def _add_bond(
        side_bonds: Dict[int, List[int]], m: Chem.Mol, mol_i: int, a: int, b: int
    ) -> None:
        bond = m.GetBondBetweenAtoms(a, b)
        if bond is None:
            return
        bi = bond.GetIdx()
        if bi not in side_bonds[mol_i]:
            side_bonds[mol_i].append(bi)

    # For each changed mapped pair, mark the corresponding bond (if present) + endpoint atoms
    for mn1, mn2 in changed_pairs:
        # reactants
        if mn1 in r_mn_loc and mn2 in r_mn_loc:
            mi1, ai1 = r_mn_loc[mn1]
            mi2, ai2 = r_mn_loc[mn2]
            if mi1 == mi2:
                m = reactants[mi1]
                _add_atom(r_atoms, mi1, ai1)
                _add_atom(r_atoms, mi1, ai2)
                _add_bond(r_bonds, m, mi1, ai1, ai2)

        # products
        if mn1 in p_mn_loc and mn2 in p_mn_loc:
            mi1, ai1 = p_mn_loc[mn1]
            mi2, ai2 = p_mn_loc[mn2]
            if mi1 == mi2:
                m = products[mi1]
                _add_atom(p_atoms, mi1, ai1)
                _add_atom(p_atoms, mi1, ai2)
                _add_bond(p_bonds, m, mi1, ai1, ai2)

    return RxnHighlights(
        r_atoms=r_atoms, r_bonds=r_bonds, p_atoms=p_atoms, p_bonds=p_bonds
    )


def _reaction_molecules(rxn: rdChemReactions.ChemicalReaction) -> List[Chem.Mol]:
    return list(rxn.GetReactants()) + list(rxn.GetAgents()) + list(rxn.GetProducts())


def _auto_canvas_size(
    rxn: rdChemReactions.ChemicalReaction,
    size: Optional[Tuple[int, int]],
    legend: Optional[str],
) -> Tuple[int, int]:
    """Choose a compact canvas so small reactions do not become tiny."""
    if size is not None:
        return size

    mols = _reaction_molecules(rxn)
    n_components = max(1, len(mols))
    n_atoms = sum(m.GetNumAtoms() for m in mols if m is not None)
    width = int(max(520, min(1500, 180 + 150 * n_components + 34 * n_atoms)))
    height = 290 if legend else 250
    return width, height


def _fallback_sub_img_size(
    canvas_size: Tuple[int, int], rxn: rdChemReactions.ChemicalReaction
) -> Tuple[int, int]:
    n_panels = max(1, len(_reaction_molecules(rxn)) + 1)
    sub_width = int(max(220, min(340, canvas_size[0] / n_panels)))
    return sub_width, canvas_size[1]


def _add_svg_title(svg_text: str, title: str, canvas_size: Tuple[int, int]) -> str:
    """Inject a centered SVG title after RDKit drawing finishes."""
    safe_title = html.escape(title)
    title_svg = (
        f'<text x="{canvas_size[0] / 2:.1f}" y="24" text-anchor="middle" '
        'font-family="sans-serif" font-size="18" font-weight="700" '
        f'fill="#1a1a1a">{safe_title}</text>'
    )
    return svg_text.replace("</svg>", f"{title_svg}</svg>")


def _add_pil_title(image: Any, title: str, canvas_size: Tuple[int, int]) -> Any:
    """Overlay a centered title on a PIL reaction image."""
    from PIL import ImageDraw, ImageFont

    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
        )
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=font)
    text_width = bbox[2] - bbox[0]
    x = max(8, (canvas_size[0] - text_width) / 2)
    draw.text((x, 8), title, fill="#1a1a1a", font=font)
    return image


def visualize_reaction(  # noqa: C901
    rsmi: str,
    *,
    size: Optional[Tuple[int, int]] = None,
    sub_img_size: Tuple[int, int] = (
        450,
        300,
    ),  # kept for compatibility; used in fallback
    svg: bool = True,
    kekulize: bool = False,
    show_atom_maps: bool = False,
    highlight_changes: bool = True,
    legend: Optional[str] = None,
    fixed_bond_length: Optional[float] = None,
    padding: float = 0.06,
) -> Union[str, Any]:  # Any covers PIL.Image.Image when Cairo is available
    """
    More visual RDKit reaction rendering.

    Improvements vs Draw.ReactionToImage:
    - Uses rdMolDraw2D for cleaner SVG/Cairo output and better control.
    - Optional highlighting of changed bonds using atom-map numbers.
    - Optional atom-map labels overlay (useful for debugging / talktorials).
    - Title/legend support.

    Notes
    -----
    - `highlight_changes=True` works best when rsmi contains atom-maps like [C:1].
    - For PNG/PIL output, your RDKit must be built with Cairo support.

    Parameters
    ----------
    rsmi : str
        Reaction SMILES / SMARTS (e.g. '[CH3:1][Br:2]>>[CH3:1][OH:2]').
    size : (w, h), optional
        Canvas size in pixels. If omitted, a compact size is inferred from
        the number of reaction components and atoms.
    svg : bool
        If True return SVG string; else return PIL image (Cairo).
    kekulize : bool
        If True kekulize molecules before drawing (sometimes nicer for aromatic).
    show_atom_maps : bool
        If True, draw atom-map numbers as labels.
    highlight_changes : bool
        If True, detect and highlight changed bonds (requires atom maps).
    legend : str | None
        Optional title at the top.
    fixed_bond_length : float, optional
        Affects perceived scale / whitespace. If omitted, a readable default
        is chosen for the inferred canvas.
    padding : float
        Relative padding around the drawing.

    Returns
    -------
    str (SVG) or PIL.Image.Image
    """
    rxn = rdChemReactions.ReactionFromSmarts(rsmi, useSmiles=True)
    if rxn is None:
        raise ValueError("Invalid reaction SMILES/SMARTS")

    rxn.Initialize()
    canvas_size = _auto_canvas_size(rxn, size, legend)
    bond_length = fixed_bond_length if fixed_bond_length is not None else 34.0

    # Ensure 2D coords
    for m in list(rxn.GetReactants()) + list(rxn.GetProducts()) + list(rxn.GetAgents()):
        if m is not None:
            _ensure_2d(m)

    # Optional kekulization (copy to be safe)
    if kekulize:

        def _kek(m: Chem.Mol) -> Chem.Mol:
            m2 = Chem.Mol(m)
            try:
                Chem.Kekulize(m2, clearAromaticFlags=True)
            except Exception:
                pass
            return m2

        rxn2 = rdChemReactions.ChemicalReaction()
        for m in rxn.GetReactants():
            rxn2.AddReactantTemplate(_kek(m))
        for m in rxn.GetAgents():
            rxn2.AddAgentTemplate(_kek(m))
        for m in rxn.GetProducts():
            rxn2.AddProductTemplate(_kek(m))
        rxn2.Initialize()
        rxn = rxn2

    # Highlights
    hl = _find_changed_bonds_by_atommap(rxn) if highlight_changes else None
    # rdMolDraw2D expects a single highlight dict; for reactions, DrawReaction accepts
    # per-mol highlights in newer RDKit builds. We'll attempt that; otherwise fallback.
    # (Fallback still gives improved aesthetics via Draw.ReactionToImage.)
    try:
        if svg:
            drawer = rdMolDraw2D.MolDraw2DSVG(canvas_size[0], canvas_size[1])
        else:
            drawer = rdMolDraw2D.MolDraw2DCairo(canvas_size[0], canvas_size[1])

        opts = drawer.drawOptions()
        opts.fixedBondLength = bond_length
        opts.padding = padding
        opts.continuousHighlight = True
        opts.highlightBondWidthMultiplier = 18
        opts.useBWAtomPalette()  # crisp, publication-ish defaults

        if show_atom_maps:
            # draw atom-map numbers
            opts.atomLabels = {}  # type: ignore[attr-defined]
            for m in (
                list(rxn.GetReactants())
                + list(rxn.GetAgents())
                + list(rxn.GetProducts())
            ):
                for a in m.GetAtoms():
                    mn = a.GetAtomMapNum()
                    if mn > 0:
                        opts.atomLabels[(m, a.GetIdx())] = str(
                            mn
                        )  # may be ignored in some builds

        # Per-molecule highlights (reactants/products only; agents usually ignored)
        if hl is not None:
            # Build per-template highlight specs
            # Newer RDKit supports passing these directly to DrawReaction.
            drawer.DrawReaction(
                rxn,
                highlightByReactant=False,
                highlightReactantAtoms=hl.r_atoms,
                highlightReactantBonds=hl.r_bonds,
                highlightProductAtoms=hl.p_atoms,
                highlightProductBonds=hl.p_bonds,
            )
        else:
            drawer.DrawReaction(rxn, highlightByReactant=False)

        drawer.FinishDrawing()
        if svg:
            svg_text = drawer.GetDrawingText()
            return _add_svg_title(svg_text, legend, canvas_size) if legend else svg_text
        else:
            # Cairo returns PNG bytes
            from PIL import Image
            import io

            png = drawer.GetDrawingText()
            image = Image.open(io.BytesIO(png))
            return _add_pil_title(image, legend, canvas_size) if legend else image

    except Exception:
        # Safe fallback (still decent) if DrawReaction signature differs in your RDKit build
        from rdkit.Chem import Draw as _Draw

        fallback = _Draw.ReactionToImage(
            rxn,
            subImgSize=(
                _fallback_sub_img_size(canvas_size, rxn)
                if size is None
                else sub_img_size
            ),
            useSVG=svg,
        )
        if legend and svg:
            return _add_svg_title(fallback, legend, canvas_size)
        if legend:
            return _add_pil_title(fallback, legend, canvas_size)
        return fallback
