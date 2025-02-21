from typing import List, Dict
from IPython.display import display, HTML, SVG
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdChemReactions


class ChemicalReactionVisualizer:
    @staticmethod
    def create_html_table_with_svgs(
        svg_list: List[str],
        titles: List[str],
        num_cols: int = 2,
        orientation: str = "vertical",
        title_size: int = 16,
    ) -> HTML:
        """
        Creates an HTML table to display SVG images with titles in
        a structured 'subplot-like' layout.

        Parameters:
        - svg_list (List[str]): List of SVG content strings.
        - titles (List[str]): Corresponding titles for each SVG image.
        - num_cols (int): Defines the number of columns for the
                        'vertical' layout or rows for 'horizontal' layout.
        - orientation (str): Layout orientation of images ('vertical' or 'horizontal').
        - title_size (int): Font size of the titles displayed above each image.

        Returns:
        - HTML: HTML object to be displayed within an IPython notebook environment.
        """
        html = "<table>"
        title_style = f"font-size:{title_size}px;"  # CSS to control title size
        if orientation == "vertical":
            for i in range(0, len(svg_list), num_cols):
                html += "<tr>"
                for j in range(num_cols):
                    if i + j < len(svg_list):
                        html += f"<td style='border:1px solid black; padding:10px'><b style='{title_style}'>{titles[i+j]}</b><br>{svg_list[i+j]}</td>"
                html += "</tr>"
        else:
            for j in range(num_cols):
                html += "<tr>"
                for i in range(j, len(svg_list), num_cols):
                    html += f"<td style='border:1px solid black; padding:10px'><b style='{title_style}'>{titles[i]}</b><br>{svg_list[i]}</td>"
                html += "</tr>"
        html += "</table>"
        return HTML(html)

    @staticmethod
    def visualize_reaction(
        reaction_smiles: str,
        img_size: tuple = (600, 200),
        highlight_by_reactant: bool = True,
        mol_scale: float = 0.9,
        bond_line_width: float = 2.0,
        atom_label_font_size: int = 12,
        padding: float = 0.01,
        show_atom_map: bool = False,
    ) -> SVG:
        """
        Visualizes a chemical reaction using RDKit and generates an SVG image for display.

        Parameters:
        - reaction_smiles (str): SMILES string representing the chemical reaction.
        - img_size (tuple): Dimensions of the output image (width, height).
        - highlight_by_reactant (bool): Whether to highlight reactants in the image.
        - mol_scale (float): Scale factor for the size of the molecules in the image.
        - bond_line_width (float): Line width for the bonds in the drawing.
        - atom_label_font_size (int): Font size for the atom labels in the drawing.
        - padding (float): Padding around the image in the SVG.
        - show_atom_map (bool): Whether to display atom mapping numbers on the atoms.

        Returns:
        - SVG: An SVG object containing the rendered reaction.
        """
        reaction = rdChemReactions.ReactionFromSmarts(reaction_smiles, useSmiles=True)
        rdChemReactions.PreprocessReaction(reaction)
        if show_atom_map:
            for mol in list(reaction.GetReactants()) + list(reaction.GetProducts()):
                for atom in mol.GetAtoms():
                    if atom.HasProp("molAtomMapNumber"):
                        atom.SetProp("atomLabel", atom.GetProp("molAtomMapNumber"))

        drawer = rdMolDraw2D.MolDraw2DSVG(img_size[0], img_size[1])
        opts = drawer.drawOptions()
        opts.scale = mol_scale
        opts.bondLineWidth = bond_line_width
        opts.atomLabelFontSize = atom_label_font_size
        opts.padding = padding

        drawer.DrawReaction(reaction, highlightByReactant=highlight_by_reactant)
        drawer.FinishDrawing()
        return SVG(drawer.GetDrawingText())

    @staticmethod
    def visualize_and_compare_reactions(
        input_dict: Dict[str, str],
        id_col: str = "R-id",
        img_size: tuple = (1000, 600),
        num_cols: int = 2,
        orientation: str = "vertical",
        show_atom_map: bool = False,
    ) -> None:
        """
        Visualizes and compares multiple chemical reactions,
        displaying them side by side in an HTML table.

        Parameters:
        - input_dict (Dict[str, str]): Dictionary with reaction
            identifiers as keys and SMILES strings as values.
        - id_col (str): A dictionary key to exclude
                        from visualization (typically metadata).
        - img_size (tuple): size if image.
        - num_cols (int): Number of columns in the display table.
        - orientation (str): vertically or horizontally.
        """
        svg_list = []
        titles = []
        for key, reaction_str in input_dict.items():
            if key != id_col:
                svg = ChemicalReactionVisualizer.visualize_reaction(
                    reaction_str,
                    img_size=img_size,
                    highlight_by_reactant=True,
                    show_atom_map=show_atom_map,
                )
                svg_list.append(svg.data)
                titles.append(key)
        display(
            ChemicalReactionVisualizer.create_html_table_with_svgs(
                svg_list, titles, num_cols, orientation
            )
        )
