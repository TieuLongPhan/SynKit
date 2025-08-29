from __future__ import annotations
from typing import Tuple, List, Dict, Any

# === Motif definitions ===
motif_1 = [
    "Source.L>>L",
    "L>>M",
    "M>>N",
    "N>>O",
    "O>>P",
    "P>>Q",
    "Q>>OutputPool_Q",
    "OutputPool_Q>>Exported_Q",
    "Exported_Q>>Removed",
]
motif_1_config = {
    "sources": {"Source.L": "rate_limited"},
    "sinks": {"Removed": "immediate_sink"},
}

motif_2 = [
    "Source.A>>A",
    "A.B>>C",
    "C>>D",
    "D.E>>F",
    "F>>G",
    "G>>H",
    "H>>I",
    "I>>OutputPool_I",
    "OutputPool_I>>Removed",
]
motif_2_config = {
    "sources": {"Source.A": "limited", "Source.B": "limited"},
    "sinks": {"Removed": "immediate_sink"},
}

motif_3 = [
    "Source.P>>P",  # input of precursor P
    "P>>Q",  # P produces Q
    "P>>R",  # P also produces R (separate event)
    "Q>>S",  # Q -> S
    "R>>T",  # R -> T
    "S.T>>U",  # S + T -> U (requires both S and T present)
    "U>>V",
    "V>>W",
    "W>>Waste",
    "Waste>>Removed",
]
motif_3_config = {
    "regime": "stochastic",
    "sources": {"Source.P": {"type": "limited", "initial": 50}},
    "sinks": {"Removed": {"type": "immediate_sink"}},
}


motif_4 = [
    "Source.S>>S",
    "S>>U",
    "S>>V",
    "U>>W",
    "V>>X",
    "W.X>>Y",
    "Y>>Z",
    "Z>>OutputPool_Z",
    "OutputPool_Z>>Removed",
]
motif_4_config = {
    "sources": {"Source.S": {"type": "unlimited"}},
    "sinks": {"Removed": {"type": "immediate_sink"}},
}


motif_5 = [
    "Source.A>>A",
    "Source.B>>B",
    "Source.C>>C",
    "A.C>>G",
    "B.C>>G",
    "G>>H",
    "H.I>>J",
    "I>>K",
    "J.K>>L",
    "L>>OutputPool_L",
    "OutputPool_L>>Removed",
]
motif_5_config = {
    "sources": {"Source.A": "limited", "Source.B": "limited", "Source.C": "limited"},
    "sinks": {"Removed": "immediate_sink"},
}

motif_6 = [
    "Source.Seed>>Seed",
    "Source.X>>X",
    "Seed.X>>X.Seed",
    "X.Seed>>X.X",
    "X>>Y",
    "Y>>Z",
    "Z>>Waste",
    "X>>X_inactive",
    "X_inactive.RepairFactor>>X_repair",
    "X_repair>>X",
    "X_inactive>>Waste",
    "Waste>>Removed",
]
motif_6_config = {
    "sources": {"Source.Seed": "limited", "Source.X": "limited"},
    "sinks": {"Removed": "immediate_sink"},
}

motif_7 = [
    "Source.A>>A",
    "Cat.A>>Cat_A",
    "Cat_A>>Cat_A*",
    "Cat_A*>>Cat_B",
    "Cat_B>>Cat.B",
    "B>>C",
    "Cat.Inhibitor>>Cat_Inhibitor",
    "Cat_Inhibitor>>Cat",
]
motif_7_config = {"sources": {"Source.A": "limited"}, "sinks": {"C": "export_after_T"}}

motif_8 = [
    "Source.A>>A",
    "A>>B",
    "B>>C",
    "C>>D",
    "D.C>>C_amplified",
    "C_amplified>>C",
    "C>>E",
    "E>>F",
    "F>>Waste",
    "Waste>>Removed",
]
motif_8_config = {
    "sources": {"Source.A": "rate_limited"},
    "sinks": {"Removed": "immediate_sink"},
}

motif_9 = [
    "Source.X>>X",
    "X>>Y",
    "X>>Z",
    "Y.X>>Z_enhanced",
    "Z_enhanced>>Z",
    "Z>>W",
    "W>>OutputPool",
    "OutputPool>>Exported",
]
motif_9_config = {
    "sources": {"Source.X": "rate_limited"},
    "sinks": {"Exported": "immediate_sink"},
}

motif_10 = [
    "Source.Rinit>>Rinit",
    "Rinit>>R",
    "R>>R.S",
    "R.S>>R.R",
    "R>>P1",
    "R>>P2",
    "P1>>Waste",
    "P2>>Waste",
    "Waste>>Removed",
]
motif_10_config = {
    "sources": {"Source.Rinit": "limited"},
    "sinks": {"Removed": "immediate_sink"},
}

motif_11 = [
    "Source.A>>A",
    "Source.B>>B",
    "A.B>>C",
    "C>>A.B",
    "C>>D",
    "D>>E",
    "E>>C",
    "C>>F",
    "F>>OutputPool_F",
    "OutputPool_F>>Removed",
]
motif_11_config = {
    "sources": {"Source.A": "limited", "Source.B": "limited"},
    "sinks": {"Removed": "immediate_sink"},
}

motif_12 = [
    "Source.G>>G",
    "G>>G6P",
    "G6P>>6G",
    "6G>>Ru5P.CO2",
    "CO2>>CO2_Exported",
    "CO2_Exported>>Removed",
    "Ru5P>>R5P",
    "R5P>>Biomass",
    "Biomass>>Biomass_pool",
]
motif_12_config = {
    "sources": {"Source.G": "rate_limited"},
    "sinks": {"Removed": "immediate_sink"},
}

motif_13 = [
    "Source.A>>A",
    "A>>B",
    "B>>C",
    "C>>D",
    "D>>A",
    "A>>E",
    "E>>Waste",
    "Waste>>Removed",
]
motif_13_config = {
    "sources": {"Source.A": "rate_limited"},
    "sinks": {"Removed": "immediate_sink"},
}

motif_14 = [
    "Source.A_cyto>>A_cyto",
    "A_cyto>>A_mito",
    "A_mito>>B_mito",
    "B_mito>>C_mito",
    "C_mito>>C_cyto",
    "C_cyto>>D",
    "D>>Exported_D",
    "Exported_D>>Removed",
]
motif_14_config = {
    "sources": {"Source.A_cyto": "rate_limited"},
    "sinks": {"Removed": "immediate_sink"},
}

motif_15 = [
    "Source.Init>>Init",
    "Source.M>>M",
    "Init>>R",
    "R.M>>RM",
    "RM.M>>RMM",
    "RMM.M>>RMMM",
    "RMM>>Branch_site",
    "RMMM>>P",
    "P>>Processed",
    "Processed>>Removed",
]
motif_15_config = {
    "sources": {"Source.Init": "limited", "Source.M": "rate_limited"},
    "sinks": {"Removed": "immediate_sink"},
}

motif_16 = [
    "Xu5P.R5P>>S7P.GAP",
    "S7P.GAP>>F6P.E4P",
    "F6P>>F1P",
    "E4P>>E3P",
    "E3P.R>>Xfr",
    "Xfr>>Glyco",
    "Glyco>>OutputPool_Glyco",
    "OutputPool_Glyco>>Removed",
]
motif_16_config = {
    "sources": {"Xu5P": "limited"},
    "sinks": {"Removed": "immediate_sink"},
}

motif_17 = [
    "Source.A>>A",
    "A>>B",
    "A>>B",
    "B>>C",
    "C>>D",
    "D>>E",
    "E>>Outcome",
    "Outcome>>Removed",
]
reaction_meta_17 = {
    1: {"id": "r17_a_to_b_E1", "enzyme": "E1"},
    2: {"id": "r17_a_to_b_E2", "enzyme": "E2"},
}
motif_17_config = {
    "sources": {"Source.A": "limited"},
    "sinks": {"Removed": "immediate_sink"},
}

# biochemical examples
glutamine_pathway = [
    "Source.Gln>>Gln",
    "Gln.Enz_gls>>Enz_gls:Gln",
    "Enz_gls:Gln.H2O>>Enz_gls:Glu_intermediate.NH3",
    "Enz_gls:Glu_intermediate>>Enz_gls.Glu.NH3",
    "NH3.H+>>NH4",
    "NH4>>Exported_NH4",
    "Exported_NH4>>Removed",
]
glutamine_pathway_config = {
    "sources": {"Source.Gln": "limited"},
    "sinks": {"Removed": "immediate_sink"},
}

glutamate_deamination = [
    "Source.Glu>>Glu",
    "Glu.Enz_GDH>>Enz_GDH:Glu",
    "Enz_GDH:Glu.NAD+>>Enz_GDH:Imine.NADH.H+",
    "Enz_GDH:Imine.H2O>>Enz_GDH:alphaKG.NH3",
    "Enz_GDH:alphaKG>>Enz_GDH.alphaKG",
    "NH3.H+>>NH4",
    "NH4>>Exported_NH4",
    "Exported_NH4>>Removed",
    "NADH>>NAD+_respired",
]
glutamate_deamination_config = {
    "sources": {"Source.Glu": "limited"},
    "sinks": {"Removed": "immediate_sink"},
}

cps_direct = [
    "Source.NH3>>NH3",
    "Source.HCO3>>HCO3",
    "HCO3.ATP.Enz_CPS>>Enz_CPS:carboxyphosphate.ADP",
    "Enz_CPS:carboxyphosphate.NH3>>Enz_CPS:carbamate.Pi",
    "Enz_CPS:carbamate.ATP>>Enz_CPS:carbamoylP.ADP",
    "Enz_CPS:carbamoylP>>Enz_CPS.CarbamoylP",
    "CarbamoylP>>OutputPool_CarbamoylP",
    "OutputPool_CarbamoylP>>Exported_CarbamoylP",
    "Exported_CarbamoylP>>Removed",
]
cps_direct_config = {
    "sources": {"Source.NH3": "rate_limited", "Source.HCO3": "rate_limited"},
    "sinks": {"Removed": "immediate_sink"},
}

cps_glutamine_channel = [
    "Source.Gln>>Gln",
    "Gln.Enz_CPS>>Enz_CPS:Gln",
    "Enz_CPS:Gln.H2O>>Enz_CPS:Glu.NH3_channel",
    "HCO3.ATP.Enz_CPS>>Enz_CPS:carboxyphosphate.ADP",
    "Enz_CPS:carboxyphosphate.NH3_channel>>Enz_CPS:carbamate.Pi",
    "Enz_CPS:carbamate.ATP>>Enz_CPS:carbamoylP.ADP",
    "Enz_CPS:carbamoylP>>Enz_CPS.CarbamoylP",
    "CarbamoylP>>OutputPool_CarbamoylP",
    "OutputPool_CarbamoylP>>Exported_CarbamoylP",
    "Exported_CarbamoylP>>Removed",
]
cps_glutamine_channel_config = {
    "sources": {"Source.Gln": "limited"},
    "sinks": {"Removed": "immediate_sink"},
}

# Pack them into a dict for iteration
ALL_MOTIFS: Dict[str, Tuple[List[str], Dict[str, Any]]] = {
    "motif_1": (motif_1, motif_1_config),
    "motif_2": (motif_2, motif_2_config),
    "motif_3": (motif_3, motif_3_config),
    "motif_4": (motif_4, motif_4_config),
    "motif_5": (motif_5, motif_5_config),
    "motif_6": (motif_6, motif_6_config),
    "motif_7": (motif_7, motif_7_config),
    "motif_8": (motif_8, motif_8_config),
    "motif_9": (motif_9, motif_9_config),
    "motif_10": (motif_10, motif_10_config),
    "motif_11": (motif_11, motif_11_config),
    "motif_12": (motif_12, motif_12_config),
    "motif_13": (motif_13, motif_13_config),
    "motif_14": (motif_14, motif_14_config),
    "motif_15": (motif_15, motif_15_config),
    "motif_16": (motif_16, motif_16_config),
    "motif_17": (motif_17, motif_17_config),
    "glutamine_pathway": (glutamine_pathway, glutamine_pathway_config),
    "glutamate_deamination": (glutamate_deamination, glutamate_deamination_config),
    "cps_direct": (cps_direct, cps_direct_config),
    "cps_glutamine_channel": (cps_glutamine_channel, cps_glutamine_channel_config),
}
