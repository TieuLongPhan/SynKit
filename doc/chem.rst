.. _chem:

Chem
====

The `synkit.Chem` module provides tools for reaction SMILES processing, including atom‐map canonicalization, equivalence validation, and SMILES standardization.

Canonicalization
----------------
The class :py:class:`~synkit.Chem.Reaction.canon_rsmi.CanonRSMI` standardizes reaction SMILES and atom-map indices by computing a canonical relabeling of mapped atoms. It employs a Weisfeiler–Lehman colour-refinement back-end (default: **3** iterations) to ensure that each atom-map assignment is uniquely and consistently ordered across isomorphic reactions :cite:`weisfeiler1968reduction`.

.. code-block:: python

   from synkit.Chem.Reaction import CanonRSMI

   canon = CanonRSMI(backend='wl', wl_iterations=3)
   canon.canonicalise(
       '[CH3:1][CH:2]=[O:3].[CH:4]([H:7])([H:8])[CH:5]=[O:6]'
       '>>'
       '[CH3:1][CH:2]=[CH:4][CH:5]=[O:6].[O:3]([H:7])([H:8])'
   )
   print(canon.canonical_rsmi)
   >> '[CH:3]([CH3:7])=[O:8].[H:1][CH:4]([H:2])[CH:6]=[O:5]>>[CH:3](=[CH:4][CH:6]=[O:5])[CH3:7].[H:1][O:8][H:2]'

AAM comparison
--------------
The class :py:class:`~synkit.Chem.Reaction.aam_validator.AAMValidator` verifies atom‐map equivalence by constructing an Imaginary Transition State (ITS) graph for each reaction and testing graph isomorphism via NetworkX’s VF2 algorithm. This approach ensures that two atom‐mapped reactions produce identical ITS topologies before and after mapping :cite:`phan2025syntemp`.

.. code-block:: python

   from synkit.Chem.Reaction import AAMValidator

   validator = AAMValidator()
   rsmi_1 = (
       '[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][OH:6]'
       '>>'
       '[CH3:1][C:2](=[O:3])[O:6][CH3:5].[OH2:4]'
   )
   rsmi_2 = (
       '[CH3:5][C:1](=[O:2])[OH:3].[CH3:6][OH:4]'
       '>>'
       '[CH3:5][C:1](=[O:2])[O:4][CH3:6].[OH2:3]'
   )
   is_eq = validator.smiles_check(rsmi_1, rsmi_2, check_method='ITS')
   print(is_eq)
   >> True

Standardization
---------------
The class :py:class:`~synkit.Chem.Reaction.standardize.Standardize` cleans and normalizes reaction SMILES by applying RDKit sanitization, removing atom‐map annotations, and stripping stereochemical labels. Its configurable options allow you to toggle atom‐map removal and stereo‐ignoring treatments to produce a minimal, canonical SMILES representation suitable for downstream processing.```

.. code-block:: python

   from synkit.Chem.Reaction.standardize import Standardize

   std = Standardize()
   rsmi = (
       '[CH3:1][CH:2]=[O:3].[CH:4]([H:7])([H:8])[CH:5]=[O:6]'
       '>>'
       '[CH3:1][CH:2]=[CH:4][CH:5]=[O:6].[O:3]([H:7])([H:8])'
   )
   std_rsmi = std.fit(rsmi, remove_aam=True, ignore_stereo=True)
   print(std_rsmi)
   >> 'CC=O.CC=O>>CC=CC=O.O'

See Also
--------

- :mod:`synkit.Graph` — graph modeling  