# ChiralFinder RotA

This directory vendors the exact `RotA.xlsx` file from ChiralFinder revision
`6aa14678f0f0d559badf2dac89811fbfa085459e` under that repository's MIT
license. Preserve `LICENSE` and cite Shi et al., DOI
`10.1016/j.xcrp.2025.103065`.

RotA has 650 rows and labels one or more axial loci per molecule. It is not a
balanced chiral/achiral dataset: every entry was selected as axially chiral,
and its 2D SMILES generally describes the constitution rather than a specific
atropisomeric configuration. Consequently, it must not be included in the ACS
whole-molecule binary or stereo-stripped four-state accuracy totals.

It is registered for a future task-specific benchmark that predicts axial
loci (and, where sufficient conformer information is supplied, configuration).
The central report also runs the supplied and stereo-removed whole-molecule
classifiers as diagnostics only. Their output distributions and timings are
not RotA accuracy because the reference task and label space do not match.
