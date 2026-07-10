import matplotlib

matplotlib.use("Agg")


def test_molecule_namespace_exports_domain_api():
    from synkit.Vis import molecule
    from synkit.Vis.molecule import (
        draw_molecule_graph,
        draw_reaction_graphs,
        find_reaction_highlights,
    )

    assert molecule.draw_molecule_graph is draw_molecule_graph
    assert callable(draw_reaction_graphs)
    assert callable(find_reaction_highlights)


def test_its_namespace_exports_domain_api():
    from synkit.Vis import its
    from synkit.Vis.its import draw_its_from_rsmi, draw_its_graph, draw_its_only

    assert its.draw_its_graph is draw_its_graph
    assert callable(draw_its_from_rsmi)
    assert callable(draw_its_only)


def test_mtg_namespace_exports_domain_api():
    from synkit.Vis import mtg
    from synkit.Vis.mtg import draw_mtg_graph, draw_mtg_steps

    assert mtg.draw_mtg_graph is draw_mtg_graph
    assert callable(draw_mtg_steps)


def test_epd_namespace_exports_domain_api():
    from synkit.Vis import epd
    from synkit.Vis.epd import MechanismVisualizer, transitions_from_epd

    assert epd.MechanismVisualizer is MechanismVisualizer
    assert callable(transitions_from_epd)


def test_space_namespace_exports_domain_api():
    from synkit.Vis import space
    from synkit.Vis.space import Embedding, scatter_plot

    assert space.Embedding is Embedding
    assert callable(scatter_plot)


def test_reaction_namespace_exports_domain_api():
    from synkit.Vis import reaction
    from synkit.Vis.reaction import RXNVis, RuleVis

    assert reaction.RXNVis is RXNVis
    assert reaction.RuleVis is RuleVis


def test_crn_namespace_exports_domain_api():
    from synkit.Vis import crn
    from synkit.Vis.crn import CRNVisualizer

    assert crn.CRNVisualizer is CRNVisualizer
