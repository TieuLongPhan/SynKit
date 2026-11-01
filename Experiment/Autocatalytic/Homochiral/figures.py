"""Write the data tables that the pgfplots figures of the manuscript read.

The figures themselves are drawn by LaTeX in ``paper/autocatalytic/homochiral/main.tex``.
This script only reads ``results/`` and writes plain whitespace-separated
tables to ``paper/autocatalytic/homochiral/figures``. Run the experiment scripts first,
then from the repository root::

    PYTHONDONTWRITEBYTECODE=1 python Experiment/Autocatalytic/Homochiral/figures.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import contourpy
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE/"results"
FIGURES = HERE.parents[2]/"paper"/"autocatalytic"/"homochiral"/"figures"


def read_csv(name):
    with (RESULTS/name).open() as handle:
        return [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(handle)]


def write_table(name, header, rows):
    lines = [" ".join(header)]
    lines += [" ".join(f"{v:.6g}" for v in row) for row in rows]
    (FIGURES/name).write_text("\n".join(lines)+"\n")


def archive_tables():
    """Model sweep and every tenth archived point, for both mechanisms."""
    for mechanism in ("constructive", "destructive"):
        rows = read_csv(f"reproduction_{mechanism}.csv")
        write_table(f"archive_{mechanism}.dat", ("ratio", "model", "archived"),
                    [(r["ratio"], r["model_ee"], r["archived_ee"]) for r in rows])
        write_table(f"archive_{mechanism}_points.dat", ("ratio", "archived"),
                    [(r["ratio"], r["archived_ee"]) for r in rows[::10]])


def scan_tables():
    """Sector abscissae along the archived scans."""
    lines = json.loads((RESULTS/"scan_lines.json").read_text())
    for mechanism, record in lines.items():
        write_table(f"scan_{mechanism}.dat", ("ratio", "even", "odd", "pairsum"),
                    record["rows"])


def spectra_table():
    """Real parts of the sector spectra at the racemates of the minimal supports.

    Rows: 1010 even, odd, pair block (y = 6, 5, 4) and 0110 (y = 2.5, 1.5, 0.5).
    """
    stored = json.loads((RESULTS/"couplings.json").read_text())["supports"]
    picks = (("constructive", "1010", (6, 5, 4)), ("destructive", "0110", (2.5, 1.5, 0.5)))
    rows, positive = [], []
    for mechanism, bits, ys in picks:
        record = next(r for r in stored[mechanism] if r["bits"] == bits)["racemate"]
        for name, y in zip(("even", "odd", "pair_sum"), ys):
            for value in record["spectrum"][name]:
                rows.append((value, y, int(value > 0)))
                if value > 0:
                    positive.append((value, y))
    write_table("spectra.dat", ("re", "row", "unstable"), [r for r in rows if not r[2]])
    write_table("spectra_unstable.dat", ("re", "row"), positive)


def trajectory_table():
    """Selective-removal control: the two trajectories in the (L, D) plane."""
    rows = read_csv("control_b_trajectory.csv")
    write_table("removal.dat", ("Lplus", "Dplus", "Lminus", "Dminus"),
                [(r["L_plus"], r["D_plus"], r["L_minus"], r["D_minus"]) for r in rows])


def map_tables():
    """Boundary of the chiral instability for positive beta, and the beta = 0 row."""
    maps = json.loads((RESULTS/"maps.json").read_text())
    for mechanism, record in maps.items():
        ratio, beta = np.array(record["ratio"]), np.array(record["beta"])
        label = np.array(record["label"])
        odd = np.array(record["alpha_odd"], dtype=float)
        positive = beta > 0
        generator = contourpy.contour_generator(ratio, np.log10(beta[positive]),
                                                odd[positive])
        lines = generator.lines(0.0)
        curve = max(lines, key=len)
        curve = curve[np.argsort(curve[:, 1])]
        boundary = [(x, 10**y) for x, y in curve]
        write_table(f"map_{mechanism}.dat", ("ratio", "beta"), boundary)
        # Closed polygon of the unstable region, bounded by the right edge.
        right, low = ratio[-1], beta[positive][0]
        region = boundary+[(right, boundary[-1][1]), (right, low)]
        write_table(f"map_{mechanism}_region.dat", ("ratio", "beta"), region)
        unstable_zero = ratio[label[0] == 2]
        inactive = ratio[(label == 0).all(0)]
        summary = {"zero_onset": float(unstable_zero[0]),
                   "cores_zero": record["cores"]["zero"],
                   "cores_positive": record["cores"]["positive"],
                   "inactive_until": float(inactive[-1]) if len(inactive) else None}
        (RESULTS/f"map_{mechanism}_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(mechanism, summary, len(curve))


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    archive_tables()
    scan_tables()
    spectra_table()
    trajectory_table()
    map_tables()


if __name__ == "__main__":
    main()
