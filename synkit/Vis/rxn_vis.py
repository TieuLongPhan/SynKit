from __future__ import annotations

import io
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union

from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions, rdmolfiles
from rdkit.Chem.Draw import rdMolDraw2D


class RXNVis:
    """Visualize molecules and reactions as PIL images.

    When ``highlight_reaction_center=True`` and the input is an atom-mapped
    reaction SMILES, each molecule is drawn individually so that changed atoms
    and bonds can be color-coded:

    * **Broken bonds** (present in reactants only) — ``rc_broken_color``
    * **Formed bonds** (present in products only) — ``rc_formed_color``
    * **Order-changed bonds** (present on both sides with different order) — ``rc_atom_color``
    * **Reaction-center atoms** (any atom involved in the above) — ``rc_atom_color``

    Reactions without atom mapping fall back to the standard ``DrawReaction``
    layout automatically.

    :param width: Canvas width in pixels.
    :type width: int
    :param height: Canvas height in pixels.
    :type height: int
    :param dpi: DPI scaling factor (72 = 1×, i.e. no scaling).
    :type dpi: int
    :param background_colour: RGBA background color, each channel in [0, 1].
        Defaults to opaque white.
    :type background_colour: Optional[Tuple[float, float, float, float]]
    :param highlight_by_reactant: Color each reactant differently in the
        standard (non-RC) ``DrawReaction`` path.
    :type highlight_by_reactant: bool
    :param bond_line_width: Bond line width in points.
    :type bond_line_width: float
    :param atom_label_font_size: Base font size for atom labels (maps to
        ``MolDrawOptions.baseFontSize``).
    :type atom_label_font_size: int
    :param show_atom_map: Overlay atom-map numbers as atom labels.
    :type show_atom_map: bool
    :param highlight_reaction_center: When ``True``, detect and highlight the
        reaction center in atom-mapped reaction SMILES.
    :type highlight_reaction_center: bool
    :param rc_atom_color: RGB color for reaction-center atoms and
        order-changed bonds (values in [0, 1]).
    :type rc_atom_color: Tuple[float, float, float]
    :param rc_broken_color: RGB color for bonds broken in the reaction.
    :type rc_broken_color: Tuple[float, float, float]
    :param rc_formed_color: RGB color for bonds formed in the reaction.
    :type rc_formed_color: Tuple[float, float, float]

    .. code-block:: python

        from synkit.Vis.rxn_vis import RXNVis

        vis = RXNVis(
            width=1200,
            height=400,
            highlight_reaction_center=True,
            show_atom_map=True,
        )
        rsmi = (
            "[CH3:1][C:2](=[O:3])[O:4][CH3:5].[O:6]([H:7])[H:8]"
            ">>"
            "[CH3:1][C:2](=[O:3])[O:6][H:7].[CH3:5][O:4][H:8]"
        )
        vis.save_png(rsmi, "transesterification.png")
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 450,
        dpi: int = 96,
        background_colour: Optional[Tuple[float, float, float, float]] = None,
        highlight_by_reactant: bool = True,
        bond_line_width: float = 2.0,
        atom_label_font_size: int = 12,
        show_atom_map: bool = False,
        highlight_reaction_center: bool = False,
        rc_atom_color: Tuple[float, float, float] = (1.0, 0.5, 0.0),
        rc_broken_color: Tuple[float, float, float] = (0.9, 0.2, 0.2),
        rc_formed_color: Tuple[float, float, float] = (0.0, 0.75, 0.0),
    ) -> None:
        self.width = width
        self.height = height
        self.dpi = dpi
        self.background_colour = background_colour or (1.0, 1.0, 1.0, 1.0)
        self.highlight_by_reactant = highlight_by_reactant
        self.bond_line_width = bond_line_width
        self.atom_label_font_size = atom_label_font_size
        self.show_atom_map = show_atom_map
        self.highlight_reaction_center = highlight_reaction_center
        self.rc_atom_color = rc_atom_color
        self.rc_broken_color = rc_broken_color
        self.rc_formed_color = rc_formed_color

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self, smiles: str, return_bytes: bool = False
    ) -> Union[Image.Image, bytes]:
        """Render a molecule or reaction SMILES to a PIL image or PNG bytes.

        When ``highlight_reaction_center=True`` and ``smiles`` is an
        atom-mapped reaction SMILES, the reaction center is highlighted.
        Falls back to the standard ``DrawReaction`` layout when no atom
        mapping is present.

        :param smiles: Molecule SMILES or reaction SMILES containing ``">>"``.
        :type smiles: str
        :param return_bytes: If ``True``, return raw PNG bytes instead of a
            ``PIL.Image.Image``.
        :type return_bytes: bool
        :returns: Rendered image or PNG bytes.
        :rtype: Union[PIL.Image.Image, bytes]
        :raises ValueError: If the input cannot be parsed.
        """
        if ">>" in smiles and self.highlight_reaction_center:
            img = self._render_rc(smiles)
        else:
            img = self._render_standard(smiles)

        if self.dpi != 72:
            scale = self.dpi / 72.0
            img = img.resize(
                (int(img.width * scale), int(img.height * scale)), Image.LANCZOS
            )

        if return_bytes:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        return img

    def save_png(self, smiles: str, path: str) -> None:
        """Render and save as a PNG file.

        :param smiles: Molecule or reaction SMILES.
        :type smiles: str
        :param path: Output path (should end in ``.png``).
        :type path: str
        """
        self.render(smiles, return_bytes=False).save(path, format="PNG")

    def save_pdf(self, smiles: str, path: str, resolution: float = 300.0) -> None:
        """Render and save as a single-page PDF.

        :param smiles: Molecule or reaction SMILES.
        :type smiles: str
        :param path: Output path (should end in ``.pdf``).
        :type path: str
        :param resolution: DPI metadata for the PDF.
        :type resolution: float
        """
        self.render(smiles, return_bytes=False).convert("RGB").save(
            path, format="PDF", resolution=resolution
        )

    # ------------------------------------------------------------------
    # Standard (non-RC) rendering
    # ------------------------------------------------------------------

    def _render_standard(self, smiles: str) -> Image.Image:
        """Render using RDKit's ``DrawReaction`` / ``DrawMolecule``.

        :param smiles: Molecule or reaction SMILES.
        :type smiles: str
        :returns: Rendered and cropped image.
        :rtype: PIL.Image.Image
        """
        drawer = rdMolDraw2D.MolDraw2DCairo(self.width, self.height, 0, 0)
        opts = drawer.drawOptions()
        opts.bondLineWidth = self.bond_line_width
        opts.baseFontSize = self.atom_label_font_size
        opts.setBackgroundColour(self.background_colour)
        opts.includeAtomTags = self.show_atom_map

        try:
            if ">>" in smiles:
                rxn = rdChemReactions.ReactionFromSmarts(smiles, useSmiles=True)
                rdChemReactions.PreprocessReaction(rxn)

                if self.show_atom_map:
                    for mol in list(rxn.GetReactants()) + list(rxn.GetProducts()):
                        for atom in mol.GetAtoms():
                            if atom.HasProp("molAtomMapNumber"):
                                atom.SetProp(
                                    "atomLabel", atom.GetProp("molAtomMapNumber")
                                )

                drawer.DrawReaction(rxn, self.highlight_by_reactant, None, None)
            else:
                mol = rdmolfiles.MolFromSmiles(smiles) or rdmolfiles.MolFromSmarts(
                    smiles
                )
                if mol is None:
                    raise ValueError(f"Could not parse SMILES/SMARTS: {smiles!r}")
                drawer.DrawMolecule(mol)
        finally:
            drawer.FinishDrawing()

        img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
        bbox = img.split()[-1].getbbox()
        if bbox:
            img = img.crop(bbox)
        return img

    # ------------------------------------------------------------------
    # Reaction-center detection
    # ------------------------------------------------------------------

    @staticmethod
    def _find_reaction_center(
        rsmi: str,
    ) -> Tuple[
        Set[int],
        Set[FrozenSet[int]],
        Set[FrozenSet[int]],
        Set[FrozenSet[int]],
    ]:
        """Detect reaction-center atoms and bonds from a mapped reaction SMILES.

        Compares the bond connectivity on the reactant and product sides.  Only
        bonds where **both** endpoints carry atom-map numbers are considered.

        :param rsmi: Atom-mapped reaction SMILES containing ``">>"``.
        :type rsmi: str
        :returns: A 4-tuple ``(changed_atom_maps, formed_bonds,
            broken_bonds, changed_order_bonds)`` where each bond set contains
            ``frozenset({map_a, map_b})`` pairs.
        :rtype: Tuple[Set[int], Set[FrozenSet[int]], Set[FrozenSet[int]], Set[FrozenSet[int]]]
        """
        reactants_smi, products_smi = rsmi.split(">>", 1)

        # Keep explicit H atoms that carry atom-map numbers (e.g. [H:7]) so
        # proton-transfer bonds are correctly detected.
        _h_params = Chem.SmilesParserParams()
        _h_params.removeHs = False

        def _bond_map(side_smi: str) -> Dict[FrozenSet[int], float]:
            bonds: Dict[FrozenSet[int], float] = {}
            for smi in side_smi.split("."):
                smi = smi.strip()
                if not smi:
                    continue
                mol = Chem.MolFromSmiles(smi, _h_params)
                if mol is None:
                    continue
                for bond in mol.GetBonds():
                    a = bond.GetBeginAtom().GetAtomMapNum()
                    b = bond.GetEndAtom().GetAtomMapNum()
                    if a and b:
                        bonds[frozenset({a, b})] = bond.GetBondTypeAsDouble()
            return bonds

        r_bonds = _bond_map(reactants_smi)
        p_bonds = _bond_map(products_smi)

        formed: Set[FrozenSet[int]] = set()
        broken: Set[FrozenSet[int]] = set()
        changed_order: Set[FrozenSet[int]] = set()
        changed_maps: Set[int] = set()

        for pair in set(r_bonds) | set(p_bonds):
            r_ord = r_bonds.get(pair)
            p_ord = p_bonds.get(pair)

            if r_ord is None:
                formed.add(pair)
                changed_maps.update(pair)
            elif p_ord is None:
                broken.add(pair)
                changed_maps.update(pair)
            elif abs(r_ord - p_ord) > 1e-3:
                changed_order.add(pair)
                changed_maps.update(pair)

        return changed_maps, formed, broken, changed_order

    # ------------------------------------------------------------------
    # Per-molecule drawing with highlights
    # ------------------------------------------------------------------

    def _draw_mol(
        self,
        mol: Chem.Mol,
        mol_w: int,
        mol_h: int,
        changed_maps: Set[int],
        formed: Set[FrozenSet[int]],
        broken: Set[FrozenSet[int]],
        changed_order: Set[FrozenSet[int]],
    ) -> Image.Image:
        """Draw one molecule with reaction-center highlights.

        :param mol: RDKit molecule (2D coords are computed if absent).
        :type mol: Chem.Mol
        :param mol_w: Canvas width in pixels.
        :type mol_w: int
        :param mol_h: Canvas height in pixels.
        :type mol_h: int
        :param changed_maps: Atom-map numbers of reaction-center atoms.
        :type changed_maps: Set[int]
        :param formed: Bond pairs (atom-map frozensets) formed in the reaction.
        :type formed: Set[FrozenSet[int]]
        :param broken: Bond pairs broken in the reaction.
        :type broken: Set[FrozenSet[int]]
        :param changed_order: Bond pairs whose order changed.
        :type changed_order: Set[FrozenSet[int]]
        :returns: Rendered molecule image.
        :rtype: PIL.Image.Image
        """
        if mol.GetNumConformers() == 0:
            AllChem.Compute2DCoords(mol)

        h_atoms: List[int] = []
        h_atom_colors: Dict[int, Tuple[float, float, float]] = {}
        for atom in mol.GetAtoms():
            amap = atom.GetAtomMapNum()
            if amap and amap in changed_maps:
                idx = atom.GetIdx()
                h_atoms.append(idx)
                h_atom_colors[idx] = self.rc_atom_color

        h_bonds: List[int] = []
        h_bond_colors: Dict[int, Tuple[float, float, float]] = {}
        for bond in mol.GetBonds():
            a = bond.GetBeginAtom().GetAtomMapNum()
            b = bond.GetEndAtom().GetAtomMapNum()
            if not (a and b):
                continue
            pair = frozenset({a, b})
            if pair in formed:
                color = self.rc_formed_color
            elif pair in broken:
                color = self.rc_broken_color
            elif pair in changed_order:
                color = self.rc_atom_color
            else:
                continue
            idx = bond.GetIdx()
            h_bonds.append(idx)
            h_bond_colors[idx] = color

        drawer = rdMolDraw2D.MolDraw2DCairo(mol_w, mol_h)
        opts = drawer.drawOptions()
        opts.bondLineWidth = self.bond_line_width
        opts.baseFontSize = self.atom_label_font_size
        opts.setBackgroundColour(self.background_colour)

        if self.show_atom_map:
            for atom in mol.GetAtoms():
                if atom.GetAtomMapNum():
                    atom.SetProp("atomLabel", str(atom.GetAtomMapNum()))

        drawer.DrawMolecule(
            mol,
            highlightAtoms=h_atoms or None,
            highlightAtomColors=h_atom_colors or None,
            highlightBonds=h_bonds or None,
            highlightBondColors=h_bond_colors or None,
        )
        drawer.FinishDrawing()
        return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")

    # ------------------------------------------------------------------
    # Reaction-center rendering (compose per-molecule images)
    # ------------------------------------------------------------------

    def _render_rc(self, smiles: str) -> Image.Image:
        """Render an atom-mapped reaction SMILES with reaction-center highlights.

        If no reaction center is detected (no atom mapping), falls back to
        :meth:`_render_standard`.

        :param smiles: Atom-mapped reaction SMILES.
        :type smiles: str
        :returns: Composed reaction image.
        :rtype: PIL.Image.Image
        """
        reactants_smi, products_smi = smiles.split(">>", 1)

        h_params = Chem.SmilesParserParams()
        h_params.removeHs = False

        def _parse_mols(side_smi: str) -> List[Chem.Mol]:
            mols: List[Chem.Mol] = []
            for smi in side_smi.split("."):
                smi = smi.strip()
                if not smi:
                    continue
                mol = Chem.MolFromSmiles(smi, h_params)
                if mol is not None:
                    AllChem.Compute2DCoords(mol)
                    mols.append(mol)
            return mols

        reactants = _parse_mols(reactants_smi)
        products = _parse_mols(products_smi)

        if not reactants and not products:
            raise ValueError(f"Could not parse any molecules from: {smiles!r}")

        changed_maps, formed, broken, changed_order = self._find_reaction_center(smiles)

        # Fall back to standard layout when nothing is mapped.
        if not changed_maps:
            return self._render_standard(smiles)

        # Per-molecule canvas dimensions.
        n_mols = max(1, len(reactants) + len(products))
        n_plus_r = max(0, len(reactants) - 1)
        n_plus_p = max(0, len(products) - 1)
        arrow_w = max(60, self.width // 10)
        plus_w = max(30, self.width // 20)
        total_sep_w = arrow_w + (n_plus_r + n_plus_p) * plus_w
        mol_w = max(120, (self.width - total_sep_w) // n_mols)
        mol_h = self.height

        r_imgs = [
            self._draw_mol(m, mol_w, mol_h, changed_maps, formed, broken, changed_order)
            for m in reactants
        ]
        p_imgs = [
            self._draw_mol(m, mol_w, mol_h, changed_maps, formed, broken, changed_order)
            for m in products
        ]

        return self._compose_reaction_image(r_imgs, p_imgs, arrow_w, plus_w)

    # ------------------------------------------------------------------
    # Image composition helpers
    # ------------------------------------------------------------------

    def _make_separator(self, kind: str, width: int, height: int) -> Image.Image:
        """Create a '+' or arrow separator image.

        :param kind: ``"plus"`` for a '+' label, ``"arrow"`` for a drawn arrow.
        :type kind: str
        :param width: Image width in pixels.
        :type width: int
        :param height: Image height in pixels.
        :type height: int
        :returns: Separator image.
        :rtype: PIL.Image.Image
        """
        bg = tuple(int(c * 255) for c in self.background_colour)
        img = Image.new("RGBA", (width, height), bg)
        draw = ImageDraw.Draw(img)
        cx, cy = width // 2, height // 2
        ink = (80, 80, 80, 255)

        if kind == "arrow":
            half = width * 3 // 8
            lw = max(2, height // 40)
            ah = max(6, height // 15)
            draw.line([(cx - half, cy), (cx + half, cy)], fill=ink, width=lw)
            draw.polygon(
                [(cx + half, cy), (cx + half - ah, cy - ah), (cx + half - ah, cy + ah)],
                fill=ink,
            )
        else:
            font_size = max(16, height // 8)
            try:
                from PIL import ImageFont

                font = ImageFont.load_default(size=font_size)
            except TypeError:
                from PIL import ImageFont

                font = ImageFont.load_default()
            draw.text((cx, cy), "+", fill=ink, anchor="mm", font=font)

        return img

    def _compose_reaction_image(
        self,
        r_imgs: List[Image.Image],
        p_imgs: List[Image.Image],
        arrow_w: int,
        plus_w: int,
    ) -> Image.Image:
        """Horizontally compose reactant and product images.

        Inserts ``'+'`` separators between molecules on each side and an
        arrow between the two sides.

        :param r_imgs: Per-reactant images.
        :type r_imgs: List[PIL.Image.Image]
        :param p_imgs: Per-product images.
        :type p_imgs: List[PIL.Image.Image]
        :param arrow_w: Width in pixels for the reaction arrow.
        :type arrow_w: int
        :param plus_w: Width in pixels for each '+' separator.
        :type plus_w: int
        :returns: Final composed reaction image.
        :rtype: PIL.Image.Image
        """
        all_mol_imgs = r_imgs + p_imgs
        h = max(img.height for img in all_mol_imgs) if all_mol_imgs else self.height

        pieces: List[Image.Image] = []

        for i, img in enumerate(r_imgs):
            if i > 0:
                pieces.append(self._make_separator("plus", plus_w, h))
            pieces.append(img)

        pieces.append(self._make_separator("arrow", arrow_w, h))

        for i, img in enumerate(p_imgs):
            if i > 0:
                pieces.append(self._make_separator("plus", plus_w, h))
            pieces.append(img)

        total_w = sum(p.width for p in pieces)
        bg = tuple(int(c * 255) for c in self.background_colour)
        result = Image.new("RGBA", (total_w, h), bg)
        x = 0
        for piece in pieces:
            result.paste(piece, (x, 0))
            x += piece.width

        return result
