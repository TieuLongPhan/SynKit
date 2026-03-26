from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ColorPalette:
    """
    Muted publication-style palette for CRN visualization.

    The defaults are intentionally restrained: muted species and rule colors,
    light paper-like background, and neutral label text.
    """

    background: str
    species_fill: str
    species_edge: str
    rule_fill: str
    rule_edge: str
    reactant_edge: str
    product_edge: str
    other_edge: str
    highlight_node: str
    highlight_edge: str
    label_text: str
    node_index_text: str
    legend_text: str
    title_text: str

    def with_overrides(self, **kwargs) -> "ColorPalette":
        return replace(self, **kwargs)


_PALETTES: dict[str, ColorPalette] = {
    "paper_sage": ColorPalette(
        background="#FAF9F5",
        species_fill="#D6E0D6",
        species_edge="#5F7162",
        rule_fill="#E8E2D8",
        rule_edge="#847B6D",
        reactant_edge="#748072",
        product_edge="#8B8378",
        other_edge="#A9A39A",
        highlight_node="#EFE5C8",
        highlight_edge="#9C8457",
        label_text="#35352F",
        node_index_text="#4B4A43",
        legend_text="#35352F",
        title_text="#2F2F2A",
    ),
    "stone_leaf": ColorPalette(
        background="#FBFAF6",
        species_fill="#D7E1D7",
        species_edge="#627062",
        rule_fill="#E6E0D7",
        rule_edge="#82796C",
        reactant_edge="#748074",
        product_edge="#8B8276",
        other_edge="#A7A097",
        highlight_node="#EEE2C1",
        highlight_edge="#9A8252",
        label_text="#36352F",
        node_index_text="#4B4A44",
        legend_text="#36352F",
        title_text="#302F2A",
    ),
    "soft_moss": ColorPalette(
        background="#F8F8F3",
        species_fill="#CFDDD2",
        species_edge="#566A5D",
        rule_fill="#E6E0D7",
        rule_edge="#7B766D",
        reactant_edge="#6C7B70",
        product_edge="#837C72",
        other_edge="#9FA097",
        highlight_node="#E9DFC0",
        highlight_edge="#8F7B50",
        label_text="#34352F",
        node_index_text="#474842",
        legend_text="#34352F",
        title_text="#2E2F2A",
    ),
    "linen_clay": ColorPalette(
        background="#FCFAF7",
        species_fill="#DCE1D9",
        species_edge="#667167",
        rule_fill="#E8DED2",
        rule_edge="#837769",
        reactant_edge="#7A8177",
        product_edge="#907F73",
        other_edge="#AAA194",
        highlight_node="#F0E2C2",
        highlight_edge="#9C8050",
        label_text="#393730",
        node_index_text="#4D4A43",
        legend_text="#393730",
        title_text="#322F29",
    ),
    "fog_olive": ColorPalette(
        background="#F7F8F5",
        species_fill="#D8DFD8",
        species_edge="#616D64",
        rule_fill="#E3E0D8",
        rule_edge="#7D786E",
        reactant_edge="#717B74",
        product_edge="#867F76",
        other_edge="#A3A097",
        highlight_node="#E7E0C8",
        highlight_edge="#8C7A53",
        label_text="#33342F",
        node_index_text="#484944",
        legend_text="#33342F",
        title_text="#2D2E29",
    ),
    "mineral_dust": ColorPalette(
        background="#FAF9F7",
        species_fill="#D9DDD8",
        species_edge="#6A706A",
        rule_fill="#E7E3DC",
        rule_edge="#847D73",
        reactant_edge="#787F79",
        product_edge="#898178",
        other_edge="#A7A39C",
        highlight_node="#EDE3CB",
        highlight_edge="#93815B",
        label_text="#373732",
        node_index_text="#4B4B46",
        legend_text="#373732",
        title_text="#31312C",
    ),
    "cedar_paper": ColorPalette(
        background="#F9F8F4",
        species_fill="#D2DED4",
        species_edge="#55675A",
        rule_fill="#E6DED4",
        rule_edge="#7C7368",
        reactant_edge="#69786C",
        product_edge="#81796E",
        other_edge="#9E9A92",
        highlight_node="#ECDFBC",
        highlight_edge="#8E7749",
        label_text="#31322D",
        node_index_text="#454640",
        legend_text="#31322D",
        title_text="#2B2C27",
    ),
    "ash_linen": ColorPalette(
        background="#FBFBF9",
        species_fill="#E0E3DE",
        species_edge="#6F756E",
        rule_fill="#E9E6E0",
        rule_edge="#878075",
        reactant_edge="#7B817B",
        product_edge="#8B847C",
        other_edge="#A9A49E",
        highlight_node="#EEE5CF",
        highlight_edge="#988562",
        label_text="#383833",
        node_index_text="#4C4C47",
        legend_text="#383833",
        title_text="#32322E",
    ),
    "quiet_forest": ColorPalette(
        background="#F8F8F5",
        species_fill="#D4DED3",
        species_edge="#5D6E60",
        rule_fill="#E4DED5",
        rule_edge="#7E766B",
        reactant_edge="#6E7B70",
        product_edge="#857D73",
        other_edge="#A09D95",
        highlight_node="#EADFBE",
        highlight_edge="#8E7B4F",
        label_text="#34342F",
        node_index_text="#484843",
        legend_text="#34342F",
        title_text="#2E2E29",
    ),
    "sandstone": ColorPalette(
        background="#FCFAF5",
        species_fill="#DCE1D8",
        species_edge="#687266",
        rule_fill="#E9DFD2",
        rule_edge="#877A6C",
        reactant_edge="#7B8378",
        product_edge="#928477",
        other_edge="#ADA397",
        highlight_node="#F2E3BE",
        highlight_edge="#A28450",
        label_text="#3A3731",
        node_index_text="#4E4B44",
        legend_text="#3A3731",
        title_text="#34312B",
    ),
    "slate_botanic": ColorPalette(
        background="#F7F8F6",
        species_fill="#D5DDD8",
        species_edge="#5E6C67",
        rule_fill="#E2DFD8",
        rule_edge="#78756E",
        reactant_edge="#6D7975",
        product_edge="#817C74",
        other_edge="#9FA09A",
        highlight_node="#E7E0C7",
        highlight_edge="#897754",
        label_text="#333530",
        node_index_text="#484A46",
        legend_text="#333530",
        title_text="#2D2F2B",
    ),
    "oak_stone": ColorPalette(
        background="#FAF8F4",
        species_fill="#D7E0D4",
        species_edge="#61705F",
        rule_fill="#E7DED3",
        rule_edge="#817668",
        reactant_edge="#738072",
        product_edge="#8A7E72",
        other_edge="#A39D93",
        highlight_node="#EDE0BE",
        highlight_edge="#927A49",
        label_text="#35342E",
        node_index_text="#494841",
        legend_text="#35342E",
        title_text="#2F2E28",
    ),
    "ivory_shale": ColorPalette(
        background="#FBFAF8",
        species_fill="#DDE1DD",
        species_edge="#6A716B",
        rule_fill="#E8E5E0",
        rule_edge="#827C74",
        reactant_edge="#797F7A",
        product_edge="#88827B",
        other_edge="#A8A39D",
        highlight_node="#EDE4CE",
        highlight_edge="#93825E",
        label_text="#373733",
        node_index_text="#4B4B47",
        legend_text="#373733",
        title_text="#31312D",
    ),
    "moss_quartz": ColorPalette(
        background="#F8F8F4",
        species_fill="#D1DDD5",
        species_edge="#56685D",
        rule_fill="#E5E0D8",
        rule_edge="#79756C",
        reactant_edge="#6A7870",
        product_edge="#807B72",
        other_edge="#9DA09A",
        highlight_node="#E9E1C8",
        highlight_edge="#887652",
        label_text="#32332E",
        node_index_text="#464741",
        legend_text="#32332E",
        title_text="#2C2D28",
    ),
    "graphite_sage": ColorPalette(
        background="#F9F9F7",
        species_fill="#D8DDD8",
        species_edge="#626863",
        rule_fill="#E3E1DB",
        rule_edge="#77736C",
        reactant_edge="#6F7671",
        product_edge="#7E7972",
        other_edge="#9A9A96",
        highlight_node="#E6DEC6",
        highlight_edge="#84724F",
        label_text="#2F312D",
        node_index_text="#434540",
        legend_text="#2F312D",
        title_text="#292B27",
    ),
}


def palette_names() -> list[str]:
    return sorted(_PALETTES)


def get_palette(name: str = "paper_sage", **overrides) -> ColorPalette:
    try:
        palette = _PALETTES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown palette {name!r}. Available: {', '.join(palette_names())}"
        ) from exc

    if overrides:
        return palette.with_overrides(**overrides)
    return palette
