"""
Process all rows from polar.csv (95 888 rows).

Each row carries its own arrow_code; converted arrows are computed per-row.

Produces one JSON file per electron-pushing step count:
    debug/data/mech/polar_1_step.json
    debug/data/mech/polar_2_steps.json
    ...

Input rows are still processed internally in chunks so joblib can parallelize
the conversion work.

Usage:
    python debug/generate_arrow_dataset_batch.py
    python debug/generate_arrow_dataset_batch.py \
        --batch-size 10000 --n-jobs 4 \
        --csv Data/Mech/polar.csv \
        --out-dir debug/data/mech

    # Optional: restrict to a single arrow_code for debugging
    python debug/generate_arrow_dataset_batch.py \
        --arrow-code "10,11=10,20;12=11,12"

    # Merge step files into debug/data/mech/polar.json and add number_arrow
    python debug/format_json.py --merge --clean
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

DEBUG_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEBUG_DIR.parents[0]
for path in (str(REPO_ROOT), str(DEBUG_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from process_data import convert_record, generic_convert_arrow_code  # noqa: E402

# ── helpers ──────────────────────────────────────────────────────────────────


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=1, quotechar='"', skipinitialspace=True)
    df[["SMIRKS", "arrow_code"]] = df["SMIRKS"].str.rsplit(" ", n=1, expand=True)
    return df


def _process_one(
    csv_idx: int,
    smirks: str,
    arrow_code: str,
    orbital_class: str | None,
) -> dict:
    """Process a single row; returns a result or error dict."""
    record = {
        "SMIRKS": smirks,
        "arrow_code": arrow_code,
        "orbital pair classification": orbital_class,
    }
    try:
        result = convert_record(record)
        normalized_arrow_code = result["arrow_code"]
        expected_generic = generic_convert_arrow_code(normalized_arrow_code)
        return {
            "ok": True,
            "index": csv_idx,
            "rsmi": result["expanded_rsmi"],
            "original_rsmi": result["reaction_smiles"],
            "epd": expected_generic,
            "epd_lw": result["typed_converted"],
            "arrow_code": normalized_arrow_code,
            "original_arrow_code": result.get("original_arrow_code", arrow_code),
            "orbital_class": result.get("orbital_class"),
        }
    except Exception as exc:
        return {
            "ok": False,
            "csv_index": csv_idx,
            "SMIRKS": smirks,
            "arrow_code": arrow_code,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def _process_batch(
    rows: list[tuple[int, str, str, str | None]],
    batch_idx: int,
) -> tuple[list[dict], list[dict]]:
    """Process one batch sequentially (called from a worker process).

    rows: [(csv_idx, smirks, arrow_code, orbital_class), ...]
    """
    results, errors = [], []
    n = len(rows)
    for i, (csv_idx, smirks, arrow_code, orbital_class) in enumerate(rows, start=1):
        out = _process_one(csv_idx, smirks, arrow_code, orbital_class)
        if out.pop("ok"):
            results.append(out)
        else:
            errors.append(out)
        if i % 500 == 0:
            print(f"  [batch {batch_idx}] {i}/{n}", file=sys.stderr, flush=True)
    return results, errors


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arrow-code",
        default=None,
        help="Optional: restrict to rows with this arrow_code (for debugging)",
    )
    parser.add_argument("--csv", default="Data/Mech/polar.csv")
    parser.add_argument("--out-dir", default="debug/data/mech")
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load & optionally filter ─────────────────────────────────────────────
    df = load_csv(args.csv)
    if args.arrow_code:
        target = df[df["arrow_code"] == args.arrow_code].reset_index()
        print(
            f"arrow_code={args.arrow_code!r}  matching rows={len(target)}",
            file=sys.stderr,
        )
    else:
        target = df.reset_index()
        print(f"Processing all rows: {len(target)}", file=sys.stderr)

    total = len(target)

    # ── split into batches ───────────────────────────────────────────────────
    batches: list[list[tuple[int, str, str, str | None]]] = []
    for start in range(0, total, args.batch_size):
        chunk = target.iloc[start: start + args.batch_size]
        rows = [
            (
                int(row["index"]),
                row["SMIRKS"],
                row["arrow_code"],
                row.get("orbital pair classification"),
            )
            for _, row in chunk.iterrows()
        ]
        batches.append(rows)

    print(
        f"batches={len(batches)}  batch_size={args.batch_size}  n_jobs={args.n_jobs}",
        file=sys.stderr,
    )

    # ── parallel execution ───────────────────────────────────────────────────
    batch_results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
        delayed(_process_batch)(rows, i) for i, rows in enumerate(batches)
    )

    # ── group by number of electron-pushing steps ────────────────────────────
    grouped: dict[int, list[dict]] = defaultdict(list)
    total_ok, total_err = 0, 0
    for i, (results, errors) in enumerate(batch_results):
        for result in results:
            grouped[len(result["epd"])].append(result)

        print(
            f"  batch {i:02d}: {len(results)} ok, {len(errors)} errors",
            file=sys.stderr,
        )
        total_ok += len(results)
        total_err += len(errors)

    # ── write one file per step count ────────────────────────────────────────
    for step_count in sorted(grouped):
        suffix = "step" if step_count == 1 else "steps"
        out_path = out_dir / f"polar_{step_count}_{suffix}.json"
        with open(out_path, "w") as f:
            json.dump(grouped[step_count], f, indent=2)
        print(
            f"  {step_count} {suffix}: {len(grouped[step_count])} cases → {out_path}",
            file=sys.stderr,
        )

    print(
        f"\nDone. {total_ok} cases written, {total_err} errors across "
        f"{len(grouped)} step-count file(s).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
