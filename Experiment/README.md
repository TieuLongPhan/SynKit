# SynKit experiments

This directory contains reproducible studies and validation programs built on
the public SynKit APIs. Each documented experiment states its inputs, command,
output contract, and interpretation boundary in its local README.

## Experiment index

- [CRN](CRN/README.md): chemical reaction network validation, scaling, and the
  KEGG, formose, and BioModels case studies.
- [Lewis](Lewis/README.md): Lewis-state conversion and rule-replay studies.

Only CRN and Lewis are currently included in the public repository. Other
experiment directories remain local and ignored by Git until release.

Generated outputs should remain reproducible from tracked code and documented
inputs. Large or restricted source datasets are not redistributed unless their
license permits it; local READMEs record the expected provenance instead.
