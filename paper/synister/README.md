# Synister manuscript

This directory contains the Synister manuscript source migrated from the
standalone Synister repository. **Synister remains the paper and method name**;
SynKit is now the implementation and evidence repository.

This directory is paper-only: it contains no executable Python, campaign
runner, solver, or mutable benchmark state. Public source belongs under
`synkit/Chem/Mapper`, experiment entrypoints under `scripts`, tests under
`Test/Chem/Mapper`, and mutable campaign state under `benchmark_results`.
Frozen, digest-verified manuscript evidence belongs under `evidence`.

The initial manuscript snapshot was migrated from standalone Synister commit
`900d40e` on 28 August 2026. Subsequent manuscript and method development is
tracked here so that claims, code, tests, and frozen evidence remain together.

`main.tex` is the concise 13-page journal Article. `supplementary.tex`
contains the complete production-level formal specification and proofs. The
older 38-page development manuscript is not used as the journal main text;
its Lewis-resource derivations remain preserved verbatim in
`lewis_parity.tex` and `ref/cd.tex`, while unimplemented catalog/double-coset
proposals remain archival rather than publication claims.

The Article draft is organized around three explicitly different exact
questions:

1. global enumeration at a supplied chemical distance;
2. proof of the minimal chemical distance followed by global enumeration of
   its complete solution shell; and
3. faster reference-conditioned enumeration on a declared fixed support.

The third mode must never be reported as an unrestricted global shell. LaTeX
build artifacts, including the PDF, are reproducible and ignored by the
repository.

The reported FlowER experiment optimizes heavy-atom bond-order CD. Pendant
hydrogens have an exact conditional elimination for a fixed heavy map, with a
closed score and labeled-multiplicity formula. This does **not** make “minimize
heavy CD, then lift” a global full-atom optimizer: additive full CD requires
optimizing heavy CD plus the transported hydrogen-count discrepancy. The
current release exposes the conditional summary and flow expansion; the
combined-objective branch-and-bound is a documented future implementation.

Publication figures use the colour-blind-safe visual vocabulary adapted from
`../Style`. The workflow is native TikZ in `figures/workflow.tex`; the vector
evidence panel is regenerated from frozen JSON without manual transcription:

```bash
python scripts/plot_synister_figures.py
```

The publication application is candidate-complete alternative-ITS generation.
Given a mapped reference, Synister can enumerate the global shell at its CD,
at any other supplied CD, or at the proved global minimum, then export one AAM
per exact ITS class different from the reference class. A reference may seed
the global traversal but cannot remove candidates. These outputs are
reference-relative structural alternatives for contrastive evaluation; they
are not assertions of mechanistic invalidity.
