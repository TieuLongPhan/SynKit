import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402

from synkit.IO import rsmi_to_graph, rsmi_to_its  # noqa: E402
from synkit.Vis.epd import MechanismVisualizer, transitions_from_epd  # noqa: E402


def test_transitions_from_typed_epd_lw_preserves_original_action():
    epd_lw = [
        ["LP-/Sigma+", [1], [1, 2]],
        ["Sigma-/LP+", [2, 3], [3]],
        ["Pi-/Pi+", [4, 5], [5, 6]],
    ]

    transitions = transitions_from_epd(epd_lw)

    assert [t.kind for t in transitions] == ["LP-/B+", "B-/LP+", "B-/B+"]
    assert transitions[0].data["typed_kind"] == "LP-/Sigma+"
    assert transitions[1].data["typed_kind"] == "Sigma-/LP+"
    assert transitions[2].data["typed_kind"] == "Pi-/Pi+"


def test_visualizer_accepts_raw_typed_epd_lw():
    rsmi = "[NH3:1].[CH3:2][Cl:3]>>[NH3+:1][CH3:2].[Cl-:3]"
    reactant_graph, product_graph = rsmi_to_graph(rsmi, drop_non_aam=False)
    its_graph = rsmi_to_its(rsmi, core=False, format="tuple")
    epd_lw = [
        ["LP-/Sigma+", [1], [1, 2]],
        ["Sigma-/LP+", [2, 3], [3]],
    ]

    fig, ax = MechanismVisualizer().visualize_trajectory(
        reactant_graph,
        epd_lw,
        its_graph,
        product_graph=product_graph,
        show_legend=False,
    )

    assert fig is not None
    assert ax is not None
    labels = {text.get_text() for text in ax.texts}
    assert "Product" in labels
    assert any("Electron-flow steps" in label for label in labels)
    plt.close(fig)
