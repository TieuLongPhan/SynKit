#!/usr/bin/env python3
"""Refresh the cached KEGG modules used by the ``synkit.CRN`` case study.

This is the only part of the case study that touches the network. It re-fetches
each module's reaction equations and compound names from the KEGG REST API and
rewrites ``synkit/CRN/Benchmark/data/kegg_modules.json``.

Refreshing changes the case study's numbers if KEGG has changed, so run it
deliberately and re-run the test suite afterwards — several tests assert
specific metabolites and conservation laws.

.. rubric:: Example

.. code-block:: bash

    python Experiment/CRN/refresh_kegg_cache.py
    python Experiment/CRN/refresh_kegg_cache.py --modules M00001 M00002 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synkit.CRN.Query import KEGGExtractor  # noqa: E402
from synkit.CRN.Query.kegg_parse import parse_equation  # noqa: E402

CACHE_PATH = ROOT / "synkit" / "CRN" / "Benchmark" / "data" / "kegg_modules.json"

#: Modules to cache, with the pathway name recorded alongside each.
DEFAULT_MODULES: Dict[str, str] = {
    "M00001": "Glycolysis (Embden-Meyerhof pathway), glucose => pyruvate",
    "M00307": "Pyruvate oxidation, pyruvate => acetyl-CoA",
    "M00009": "Citrate cycle (TCA cycle, Krebs cycle)",
    "M00004": "Pentose phosphate pathway (Pentose phosphate cycle)",
}


def fetch_module(extractor: KEGGExtractor, module_id: str, name: str) -> Dict[str, object]:
    """Fetch one module's equations and compound names.

    :param extractor: KEGG extractor to fetch through.
    :type extractor: KEGGExtractor
    :param module_id: KEGG module identifier.
    :type module_id: str
    :param name: Pathway name to record.
    :type name: str
    :return: Cache entry for the module.
    :rtype: Dict[str, object]
    """
    equations = extractor.get_module_equations(module_id)

    compound_ids = set()
    for equation in equations.values():
        if not equation:
            continue
        parsed = parse_equation(equation)
        for compound_id, _ in list(parsed.reactants) + list(parsed.products):
            compound_ids.add(compound_id)

    compound_names = {}
    for compound_id in sorted(compound_ids):
        compound_name = extractor.get_compound_name(compound_id)
        if compound_name:
            compound_names[compound_id] = compound_name

    return {
        "module_id": module_id,
        "name": name,
        "equations": equations,
        "compound_names": compound_names,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Fetch the modules and rewrite the cache file.

    :param argv: Command-line arguments, defaulting to ``sys.argv[1:]``.
    :type argv: Optional[Sequence[str]]
    :return: Process exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--modules",
        nargs="+",
        default=sorted(DEFAULT_MODULES),
        help="KEGG module ids to cache",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch and report without writing the cache file",
    )
    args = parser.parse_args(argv)

    extractor = KEGGExtractor()
    modules = {}
    for module_id in args.modules:
        name = DEFAULT_MODULES.get(module_id, module_id)
        entry = fetch_module(extractor, module_id, name)
        modules[module_id] = entry
        print(
            f"{module_id}: {len(entry['equations'])} reactions, "
            f"{len(entry['compound_names'])} named compounds"
        )

    payload = {
        "source": "KEGG MODULE (https://www.kegg.jp/kegg/module.html)",
        "retrieved": date.today().isoformat(),
        "note": (
            "Cached so the case study is reproducible offline. "
            "Refresh with Experiment/CRN/refresh_kegg_cache.py."
        ),
        "modules": modules,
    }

    if args.dry_run:
        print("dry run: cache not written")
        return 0

    CACHE_PATH.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {CACHE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
