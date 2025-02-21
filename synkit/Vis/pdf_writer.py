"""
This module comprises several functions adapted from the work of Klaus Weinbauer.
The original code can be found at his GitHub repository: https://github.com/klausweinbauer/FGUtils.
Adaptations were made to enhance functionality and integrate with other system components.
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import List, Callable, Union, Tuple, Optional
import tqdm


class PdfWriter:
    """
    A utility class to create PDF reports with plots from a list of figures or dynamically generated plots.

    Parameters:
    - file (str): The file name of the output PDF.
    - plot_fn (Optional[Callable], optional): Function to create a plot for a single data entry or row.
        Expected interface: `plot_fn(data_entry, axis, **kwargs)`. Default is None.
    - plot_per_row (bool, optional): If True, calls `plot_fn` for an entire row instead of individual subplots.
        Default is False.
    - max_pages (int, optional): Maximum number of pages to create. Default is 999.
    - rows (int, optional): Number of plot rows per page. Default is 7.
    - cols (int, optional): Number of plot columns per page. Default is 2.
    - pagesize (Tuple[float, float], optional): Size of a single page (in inches). Default is (21, 29.7).
    - width_ratios (Optional[List[float]], optional): Column width ratios. Default is None.
    - show_progress (bool, optional): If True, displays a progress bar using `tqdm`. Default is True.
    """

    def __init__(
        self,
        file: str,
        plot_fn: Optional[Callable] = None,
        plot_per_row: bool = False,
        max_pages: int = 999,
        rows: int = 7,
        cols: int = 2,
        pagesize: Tuple[float, float] = (21, 29.7),
        width_ratios: Optional[List[float]] = None,
        show_progress: bool = True,
    ):
        self.pdf_pages = PdfPages(file)
        self.plot_fn = plot_fn
        self.plot_per_row = plot_per_row
        self.max_pages = max_pages
        self.rows = rows
        self.cols = cols
        self.pagesize = pagesize
        self.width_ratios = width_ratios
        self.show_progress = show_progress

    def plot(self, data: Union[List[plt.Figure], List], **kwargs):
        """
        Generate plots from data or save pre-generated figures to the PDF.

        Parameters:
        - data (Union[List[matplotlib.figure.Figure], List]): Input data or list of figures.
          If a list of figures, they are saved directly. Otherwise, the `plot_fn` is called for each data entry.
        - **kwargs: Additional keyword arguments passed to `plot_fn`.

        Returns:
        - None
        """
        # Case 1: Pre-generated figures
        if all(isinstance(item, plt.Figure) for item in data):
            for fig in tqdm.tqdm(
                data, disable=not self.show_progress, desc="Saving Figures"
            ):
                self.save_figure(fig)
            return

        # Case 2: Generate plots dynamically using `plot_fn`
        if self.plot_fn is None:
            raise ValueError(
                "plot_fn must be provided when input is not a list of figures."
            )

        if not isinstance(data, list):
            raise ValueError(
                "Data must be a list or a list of matplotlib.figure.Figure."
            )

        plots_per_page = self.rows if self.plot_per_row else self.rows * self.cols
        max_plots = self.max_pages * plots_per_page
        step = max(len(data) / max_plots, 1)
        pages = int((len(data) / step + plots_per_page - 1) // plots_per_page)

        for p in tqdm.tqdm(
            range(pages), disable=not self.show_progress, desc="Generating Pages"
        ):
            fig, ax = plt.subplots(
                self.rows,
                self.cols,
                figsize=self.pagesize,
                squeeze=False,
                gridspec_kw=(
                    {"width_ratios": self.width_ratios} if self.width_ratios else None
                ),
            )
            done = False
            for r in range(self.rows):
                if self.plot_per_row:
                    _idx = int((p * self.rows + r) * step)
                    if _idx >= len(data):
                        done = True
                        break
                    self.plot_fn(data[_idx], ax[r, :], index=_idx, **kwargs)
                else:
                    for c in range(self.cols):
                        _idx = int((p * plots_per_page + r * self.cols + c) * step)
                        if _idx >= len(data):
                            done = True
                            break
                        self.plot_fn(data[_idx], ax[r, c], index=_idx, **kwargs)
            plt.tight_layout()
            self.pdf_pages.savefig(fig, bbox_inches="tight", pad_inches=1)
            plt.close(fig)
            if done:
                break

    def save_figure(self, figure: plt.Figure):
        """
        Save a pre-generated matplotlib figure directly to the PDF.

        Parameters:
        - figure (matplotlib.figure.Figure): The figure to save.

        Returns:
        - None
        """
        if not isinstance(figure, plt.Figure):
            raise ValueError("Input must be a matplotlib.figure.Figure.")
        self.pdf_pages.savefig(figure, bbox_inches="tight", pad_inches=1)

    def close(self):
        """
        Close the PDF file, ensuring all pages are written.

        Returns:
        - None
        """
        self.pdf_pages.close()
