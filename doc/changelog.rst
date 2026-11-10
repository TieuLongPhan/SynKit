Changelog
=========

.. _implementation-plan:

Implementation plan
-------------------

.. list-table:: Planned production releases
   :header-rows: 1
   :widths: 14 18 68

   * - Release
     - Status
     - Primary outcome
   * - ``1.7.0``
     - Current development version
     - Native backends and reproducible CRN studies.
   * - ``1.8.0``
     - Planned
     - Molecular stereochemistry and whole-molecule chirality.
   * - ``1.9.0``
     - Planned
     - Stable mechanism trajectory graph model and interchange.
   * - ``2.0.0``
     - Planned
     - Unified stereo-aware reaction, rewriting, and verification APIs.

Version 1.7.0
-------------

- Add exact CRN rank, conservation, deficiency and linkage-class analysis with
  Python reference and C++ backends.
- Enumerate minimal siphons, traps and non-negative integer semiflows with
  explicit search limits and structural persistence checks.
- Add ``CRNAnalysis`` to reuse prepared network data across analyses and
  compare explicitly selected backends.
- Add Python and C++ construction paths for typed implicit-hydrogen DPO rules
  and RDKit reactions, including native network expansion and hydrogen-count
  accounting. Building the native expanders requires an RDKit C++ SDK.
- Return replayable pathway firing certificates and distinguish exhaustive
  failure from state or depth limits.
- Add structural SBML interchange and KEGG network construction with explicit
  reversible-reaction expansion and currency-compound removal. SBML kinetics
  are outside the structural import contract.
- Consolidate reproducible CRN studies, input data and experiment tests under
  ``Experiment/CRN``. Include offline validation, scaling, KEGG, formose,
  hydrogen-accounting and optional BioModels and CRNT4SBML comparisons.
- Add experiment-report consistency checks and retain exact certificates and
  raw timing samples.
- Align package, Conda and native project versions at ``1.7.0``. Bundle the
  partial-rule application library alongside the other native backends and
  validate installed wheel, Conda and Docker artifacts before publication.
- Fix Windows replay timing and concurrent native-build locking.
- Raise Requests, scikit-learn and optional PyTorch minimum versions for
  security fixes. Use Pillow 13 on Python 3.11 and newer; Python 3.10 retains
  the documented Pillow 12.x vulnerabilities.
- Keep the legacy CRNT4SBML comparator in a separate environment. Its remaining
  vulnerable numerical and packaging pins are outside SynKit runtime
  dependencies and still require upstream compatibility changes.
- Share native test builds and cache verified outputs and package downloads
  in CI, with toolchain, dependency and source checks before reuse.
