import io
from typing import Optional, Tuple, Union
from PIL import Image
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdChemReactions, rdmolfiles


class RXNVis:
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
    ):
        """
        Initialize the reaction/molecule visualizer.

        Parameters
        ----------
        width, height : int
            Canvas size in pixels (pre‐DPI).
        dpi : int
            DPI scaling factor (72 DPI = 1×).
        background_colour : tuple of 4 floats, optional
            RGBA background (0–1). Defaults to opaque white.
        highlight_by_reactant : bool
            For reactions, highlight reactant molecules.
        bond_line_width : float
            Width of bond lines.
        atom_label_font_size : int
            Font size for atom labels.
        show_atom_map : bool
            Label atoms with their map numbers.
        """
        self.width = width
        self.height = height
        self.dpi = dpi
        self.background_colour = background_colour or (1.0, 1.0, 1.0, 1.0)
        self.highlight_by_reactant = highlight_by_reactant
        self.bond_line_width = bond_line_width
        self.atom_label_font_size = atom_label_font_size
        self.show_atom_map = show_atom_map

    def render(
        self, smiles: str, return_bytes: bool = False
    ) -> Union[Image.Image, bytes]:
        """
        Render a molecule or reaction SMILES to a cropped PNG.

        Parameters
        ----------
        smiles : str
            Molecule or reaction SMARTS/SMILES.  Reactions must contain '>>'.
        return_bytes : bool
            If True, return raw PNG bytes instead of a PIL.Image.

        Returns
        -------
        PIL.Image.Image or bytes
            Cropped image (or raw PNG bytes) of the molecule/reaction.
        """
        # set up drawer
        drawer = rdMolDraw2D.MolDraw2DCairo(self.width, self.height, 0, 0)
        opts = drawer.drawOptions()
        opts.bondLineWidth = self.bond_line_width
        opts.atomLabelFontSize = self.atom_label_font_size
        opts.setBackgroundColour(self.background_colour)
        opts.includeAtomTags = self.show_atom_map

        # parse & draw
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
                    raise ValueError(f"Could not parse SMILES/SMARTS: {smiles}")
                drawer.DrawMolecule(mol)
        finally:
            drawer.FinishDrawing()

        # load into PIL
        png = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(png)).convert("RGBA")

        # crop to content
        bbox = img.split()[-1].getbbox()
        if bbox:
            img = img.crop(bbox)

        # DPI scaling
        if self.dpi != 72:
            scale = self.dpi / 72.0
            img = img.resize(
                (int(img.width * scale), int(img.height * scale)), Image.LANCZOS
            )

        return png if return_bytes else img

    def save_png(self, smiles: str, path: str) -> None:
        """
        Render and save as a PNG file.

        Parameters
        ----------
        smiles : str
            Molecule or reaction SMARTS/SMILES.
        path : str
            Output filename ending in .png.
        """
        img = self.render(smiles, return_bytes=False)
        img.save(path, format="PNG")

    def save_pdf(self, smiles: str, path: str, resolution: float = 300.0) -> None:
        """
        Render and save as a single‐page PDF.

        Parameters
        ----------
        smiles : str
            Molecule or reaction SMARTS/SMILES.
        path : str
            Output filename ending in .pdf.
        resolution : float
            DPI metadata for the PDF.
        """
        img = self.render(smiles, return_bytes=False).convert("RGB")
        img.save(path, format="PDF", resolution=resolution)


# if __name__ == "__main__":
# vis = RXNVis(width=1000,
#          height=500,
#          dpi=150,
#          background_colour=(1,1,1,1),
#          highlight_by_reactant=True,
#          show_atom_map=True)

# # get the PIL image:
# smart = '[CH3:1][CH:2]=[O:3].[CH:4]([H:7])([H:8])[CH:5]=[O:6]>>[CH3:1][CH:2]=[CH:4][CH:5]=[O:6].[O:3]([H:7])([H:8])'
# img = vis.render(smart)

# # or save directly:
# vis.save_png(smart, "reaction.png")
# vis.save_pdf(smart, "reaction.pdf")
