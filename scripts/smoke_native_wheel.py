#!/usr/bin/env python3
"""Exercise the installed native package without a compiler or source checkout."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch


def check_crn_wheel(package):
    import sys
    import numpy as np
    from synkit.CRN import (
        SynCRN,
        PathwayRealizability,
        bundled_native_library,
        native_backend,
        crnt_summary,
        integer_conservation_laws,
        conserved_moieties,
        enumerate_semiflows,
        find_siphons,
        find_traps,
    )
    from synkit.CRN.backend import backend_metadata, resolve_backend
    from synkit.CRN.native_analysis import matrix_analysis, graph_symmetry

    library = bundled_native_library()
    assert library.is_relative_to(package), library
    model = SynCRN.from_reaction_strings(["A>>B", "B>>A"])
    reference = crnt_summary(model, backend="python").to_dict()
    # A wheel must select its installed library without discovering headers,
    # running a compiler, or launching a subprocess.
    with (
        patch("subprocess.run", side_effect=AssertionError("unexpected subprocess")),
        patch(
            "subprocess.check_output",
            side_effect=AssertionError("unexpected subprocess"),
        ),
    ):
        assert resolve_backend("cpp") == ("cpp", library)
        with native_backend():
            assert crnt_summary(model).to_dict() == reference
            assert integer_conservation_laws(model) == [[1, 1]]
            assert conserved_moieties(model) == [[1, 1]]
            assert len(find_siphons(model)) == len(find_traps(model)) == 1
            flows = enumerate_semiflows(np.array([[-1, 1], [1, -1]]), kind="t")
            assert flows.complete and flows.value["flows"] == [[1, 1]]
            large = enumerate_semiflows(
                np.array([[-(10**12), 0], [1, -(10**12)], [0, 1]])
            )
            assert large.complete and large.value["flows"] == [[1, 10**12, 10**24]]
            algebra = matrix_analysis(
                [[-1, 1], [1, -1]], library_path=library, kernels="both"
            )
            assert algebra["rank"] == 1 and algebra["left"] == [[1, 1]]
            symmetry = graph_symmetry(model, library_path=library)
            assert symmetry.complete and symmetry.value["automorphism_count"] == 2
            limited = graph_symmetry(model, library_path=library, max_count=1)
            assert not limited.complete and limited.value["canonical_key"] is None
            flow = {rid: 1 for rid in model.reactions}
            initial = {sid: int(s.label == "A") for sid, s in model.species.items()}
            search = PathwayRealizability().load_syncrn_and_flow(
                model, flow=flow, initial_marking=initial, species="id", reaction="id"
            )
            search.build_petri_net_from_flow()
            outcome = search.realizability_result()
            assert (
                outcome.status.value == "realizable" and len(outcome.certificate) == 2
            )
            metadata = backend_metadata()
            assert metadata["library_sha256"] and metadata["build_sha256"]
            assert metadata["native_operations"]
    assert not any(
        name == "Experiment" or name.startswith("Experiment.") for name in sys.modules
    )
    return {"bundled_library": str(library), "capabilities": metadata["capabilities"]}


def check_canon_wheel(package):
    import networkx as nx
    from synkit.Graph.Canon._native import (
        bundled_native_canon,
        canonicalize_native_graph,
    )
    from synkit.Graph.Stereo import (
        TetrahedralStereo,
        descriptor_id,
        classify_configured_stereograph_mirror,
    )
    from synkit.Graph.Stereo.native import classify_native_stereograph_mirror

    library = bundled_native_canon()
    assert library.is_relative_to(package), library
    with (
        patch("subprocess.run", side_effect=AssertionError("unexpected subprocess")),
        patch(
            "subprocess.check_output",
            side_effect=AssertionError("unexpected subprocess"),
        ),
    ):
        graph = nx.path_graph(4)
        result = canonicalize_native_graph(graph)
        renamed = canonicalize_native_graph(
            nx.relabel_nodes(graph, {v: str(v) for v in graph})
        )
        assert result.complete and result.same_canonical_graph(renamed)
        limited = canonicalize_native_graph(nx.cycle_graph(9), max_search_nodes=1)
        assert not limited.complete and limited.canonical_code is None
        stereo = nx.star_graph(4)
        nx.set_node_attributes(stereo, {v: str(v) for v in stereo}, "color")
        descriptor = TetrahedralStereo((0, 1, 2, 3, 4), 1)
        native = classify_native_stereograph_mirror(
            stereo, (descriptor,), atom_color="color"
        )
        reference = classify_configured_stereograph_mirror(
            stereo, {descriptor_id(descriptor): descriptor}, atom_color="color"
        )
        assert native.is_definitive and native.status == reference.status
    return {"bundled_library": str(library), "stereo_status": native.status.value}


def main():
    import synkit
    from synkit.Chem.Mapper import Mapper, map_reaction_implicit
    from synkit.Chem.Mapper._native.runtime import resolve_library_path

    project = Path(__file__).resolve().parents[1]
    package = Path(synkit.__file__).resolve().parent
    if package.is_relative_to(project / "synkit"):
        raise RuntimeError("Native package smoke test imported the source checkout")
    crn = check_crn_wheel(package)
    canon = check_canon_wheel(package)
    library = resolve_library_path()
    assert library.is_relative_to(package), library
    available = Mapper.native_methods(available_only=True)
    expected = {
        "slap",
        "pabs",
        "cpabs",
        "astar",
        "abb",
        "kbb",
        "edit",
        "dfs",
        "mces",
        "sepdp",
    }
    assert expected <= set(available)
    python_result = Mapper("propagate").map("CO>>CO")
    for solver in ("pabs", "cpabs", "astar", "abb", "dfs", "mces", "sepdp"):
        result = Mapper("native", solver=solver).map("CO>>CO")
        assert result.complete and result.optimality_proven, (solver, result)
        assert result.minimum_cost == 0
        assert result.mapped_reactions == python_result.mapped_reactions
    result = Mapper("slap", backend="cpp").map("CO>>CO")
    assert result.mappings and not result.optimality_proven
    quotient = Mapper("native", hydrogen_quotient=True).map("O=CC=O>>OC#CO")
    assert quotient.complete and quotient.minimum_cost == 8
    assert len(quotient.mappings) == 2
    # A nonzero minimum, both CD modes, and explicit molecular H in the input.
    for reaction in ("CCO>>COC", "[H][H].C=C>>CC"):
        minimum = map_reaction_implicit(reaction, time_limit_seconds=10)
        assert minimum.complete and minimum.minimum_cost > 0, minimum
        for mode in ("exact", "at_most"):
            shell = map_reaction_implicit(
                reaction,
                CD=minimum.minimum_cost,
                CD_mode=mode,
                time_limit_seconds=10,
            )
            assert shell.complete and shell.mapped_reactions, shell
            assert set(shell.mapped_reactions) == set(minimum.mapped_reactions)
    print(
        json.dumps(
            {
                "version": synkit.__version__,
                "bundled_library": str(library),
                "crn": crn,
                "graph_and_stereo": canon,
                "available_solvers": available,
                "checks": "installed C++ search, Python parity, H reduction, H2, minimum/exact/at-most",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
