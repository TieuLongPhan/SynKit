# M-CSA mechanism data attribution

The selected formyl-CoA transferase mechanism data are from M-CSA (Mechanism
and Catalytic Site Atlas), EMBL-EBI:
https://www.ebi.ac.uk/thornton-srv/m-csa/entry/155/ .

M-CSA states that its data are distributed under Creative Commons Attribution
4.0 International (CC BY 4.0):
https://www.ebi.ac.uk/thornton-srv/m-csa/download/ and
https://creativecommons.org/licenses/by/4.0/ .

The research adapter transforms sixteen selected Marvin drawings into explicit
Lewis-state graphs, supplied pair-arrow groups and replay certificates. It
infers unexpanded hydrogen counts from the drawing's electron/charge fields,
retains depicted protonation and truncated residue aliases, and records atom
correspondence alternatives. These are adapted data, not unmodified full-protein
structures. Original input URLs and SHA-256 values are in
`Experiment/MTG/mcsa155_protocol.json`. Original files remain outside this repo.

The independent structure observation is deposited PDB entry 2VJL:
https://doi.org/10.2210/pdb2VJL/pdb .
Deposition authors: C. L. Berthold, C. G. Toyota, N. G. J. Richards and
Y. Lindqvist. Primary citation: https://doi.org/10.1074/jbc.M709353200 .
The research report extracts the two deposited Asp169-CG / CoA-S1P covalent
connection records; it does not reassess diffraction images or electron density.

The published overlay/substrate code is executed from an external GPL-3.0
checkout of https://github.com/JuriKolcak/overlay-graphs at the pinned commit
recorded in the comparison report. No upstream source is copied into this repo.
