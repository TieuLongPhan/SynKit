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

- Align Python package metadata, the Conda recipe, and all native CMake
  projects at version ``1.7.0``.
- Include the native partial-rule application backend in platform wheels.
- Keep CRN experiment tests alongside the experiments under
  ``Experiment/CRN/tests``.
- Raise the Requests, scikit-learn and optional PyTorch minimum versions for
  security fixes, and update the CRN environment pins.
- Use Pillow 13 on Python 3.11 and newer; retain Python 3.10 with the documented
  Pillow 12.x vulnerabilities.
