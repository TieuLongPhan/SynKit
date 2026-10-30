#!/usr/bin/env python
"""Read SynKit-exported SBML files with the legacy CRNT4SBML runtime.

This worker remains compatible with Python 3.7 for CRNT4SBML's pinned
dependency stack. The parent study writes a manifest and invokes the worker
with that isolated interpreter.
"""

import argparse
import json
import platform

import crnt4sbml
import libsbml
import networkx
import numpy


def inspect_network(item):
    """Return the structural quantities exposed by CRNT4SBML."""
    network = crnt4sbml.CRNT(item["path"])
    graph = network.get_c_graph()
    species = graph.get_species()
    return {
        "name": item["name"],
        "n_species": len(species),
        "n_reactions": len(graph.get_reactions()),
        "n_complexes": len(graph.get_complexes()),
        "n_linkage_classes": len(graph.get_linkage_classes()),
        "rank": len(species) - graph.get_dim_equilibrium_manifold(),
        "deficiency": graph.get_deficiency(),
        "weakly_reversible": graph.get_if_cgraph_weakly_reversible(),
        "linkage_class_deficiencies": graph.get_linkage_classes_deficiencies(),
    }


def main():
    """Inspect every manifest entry and serialize one JSON payload."""
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("output")
    arguments = parser.parse_args()

    with open(arguments.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)

    rows = []
    for item in manifest["networks"]:
        try:
            rows.append(inspect_network(item))
        except Exception as exc:  # reported to the parent as evidence
            rows.append(
                {
                    "name": item["name"],
                    "error": "{}: {}".format(type(exc).__name__, exc),
                }
            )

    payload = {
        "environment": {
            "python": platform.python_version(),
            "crnt4sbml": getattr(crnt4sbml, "__version__", "unknown"),
            "libsbml": libsbml.getLibSBMLDottedVersion(),
            "networkx": networkx.__version__,
            "numpy": numpy.__version__,
        },
        "networks": rows,
    }
    with open(arguments.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
