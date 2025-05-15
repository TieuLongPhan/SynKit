Chem
====

The `synkit.Chem` module provides three classes for reaction atom‐map processing:

- **CanonRSMI** (:py:class:`~synkit.Chem.Reaction.canon_rsmi.CanonRSMI`)  
  Atom‐map canonicalization via a unique relabeling of mapped atoms.  
  By default it uses the Weisfeiler–Lehman refinement algorithm :cite:`weisfeiler1968reduction` with three iterations.

- **AAMValidator** (:py:class:`~synkit.Chem.Reaction.aam_validator.AAMValidator`)  
  Atom‐map equivalence validation based on comparing *Imaginary Transition State* (ITS) graphs via NetworkX’s VF2 subgraph isomorphism :cite:`phan2025syntemp`.


Canonicalization
---------------

Standardize reaction SMILES and atom‐map indices:

.. code-block:: python

   from synkit.Chem.Reaction import CanonRSMI

   canon = CanonRSMI(backend='wl', wl_iterations=3)
   canon.canonicalise(
       '[CH3:1][CH:2]=[O:3].[CH:4]([H:7])([H:8])[CH:5]=[O:6]'
       '>>'
       '[CH3:1][CH:2]=[CH:4][CH:5]=[O:6].[O:3]([H:7])([H:8])'
   )
   print(canon.canonical_rsmi)
   >> [CH:3]([CH3:7])=[O:8].[H:1][CH:4]([H:2])[CH:6]=[O:5]>>[CH:3](=[CH:4][CH:6]=[O:5])[CH3:7].[H:1][O:8][H:2]

AAM comparison
-------------

Check whether two atom‐mapped reactions are equivalent:

.. code-block:: python

   from synkit.Chem.Reaction import AAMValidator

   validator = AAMValidator(method='ITS')
   rsmi_1 = (
       '[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][OH:6]'
       '>>'
       '[CH3:1][C:2](=[O:3])[O:6][CH3:6].[OH2:4]'
   )
   rsmi_2 = (
       '[CH3:5][C:1](=[O:2])[OH:3].[CH3:6][OH:4]'
       '>>'
       '[CH3:5][C:1](=[O:2])[O:4][CH3:6].[OH2:3]'
   )
   is_eq = validator.smiles_check(rsmi_1, rsmi_2, check_method='ITS')
   print(is_eq)
   >> True

References
----------

.. bibliography:: refs.bib
   :style: unsrt
   :cited:
