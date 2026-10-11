Chiral-feedback computational pilot
==================================

This directory implements the first bounded study described by
``paper/autocatalytic/chiral_symmetry_crn_research_plan.md``. The directory
spelling ``Autocatalyic`` follows the requested experiment destination.

Run from the repository root::

    PYTHONDONTWRITEBYTECODE=1 python Experiment/Autocatalyic/run.py
    PYTHONDONTWRITEBYTECODE=1 pytest -q -o addopts='' \
        -o cache_dir=/tmp/synkit-autocatalytic-pytest \
        Experiment/Autocatalyic/test_pilot.py

Two further scripts read the stored results and do not modify ``run.py``::

    PYTHONDONTWRITEBYTECODE=1 python Experiment/Autocatalyic/sign_structure.py
    PYTHONDONTWRITEBYTECODE=1 python Experiment/Autocatalyic/paper_figures.py

``sign_structure.py`` extracts the mirror-pair Jacobian blocks at every
positive even-stable racemate, labels each support by its negative couplings
(cross, within-hand, shared), computes the producer closure used to exclude
productive cores, and writes ``results/sign_structure.json`` together with
``support_table.tex`` and ``sign_results.tex`` in the paper directory.
``paper_figures.py`` draws the four data figures ``fig_*.pdf`` of the
manuscript. ``test_sign_structure.py`` checks the sector comparison on random
matrices, the symbolic odd block, and the agreement between an empty closure
and the absence of cores.

``backends.py`` runs Autogatito and BiRNe on eight networks exported from the
same reaction lists and writes ``results/backends.json`` and
``backend_table.tex``. The tools are GPL-3.0 and are not vendored here. Clone
``hollyritch/MRChordlessCircuits`` and ``hollyritch/Bi.R.Ne`` into one
directory, build the Autogatito extension with
``python setup.py build_ext --inplace``, and run::

    AUTOCAT_BACKENDS=/path/to/clones AUTOCAT_PYTHON=/path/to/python \
        python Experiment/Autocatalyic/backends.py

The recorded run used MRChordlessCircuits revision bb88c55 and Bi.R.Ne
revision 24df599 under Python 3.12 with python-libsbml 5.21.2 and Cython
3.3.0. All eight networks agree with the linear-programming enumeration.

``parameter_domain.py`` draws two samples of 2000 log-uniform rate sets per
mechanism (the six published coefficients, then also background conversion,
reverse ligation and racemization), tests every support for a clean chiral
instability on all racemic roots, and integrates the smallest unstable
supports of every draw. It uses a process pool (``AUTOCAT_WORKERS``).

``theory_checks.py`` verifies the balance reduction and the closed-form
instability window, computes the pitchfork coefficient, checks cycle
consistency of the rates, and tests the producer closure on random networks.
``literature_models.py`` classifies four published chiral models.
``archive_sweep.py`` compiles and runs the two archived programs unmodified
(a few minutes each, cached in ``AUTOCAT_ARCHIVE_WORK``) and repeats their
sweeps with the reaction-list model. Run all of these before
``paper_figures.py``.

Build the paper after running the experiment::

    cd paper/autocatalytic
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex si.tex

The run needs the existing scientific Python environment and ``/usr/bin/gcc``.
``environment-lock.txt`` records the tested versions. C shared libraries are
temporary files under /tmp. Matplotlib uses a /tmp configuration directory.
No production SynKit source files are changed by this experiment.

Completed calculations
----------------------

* Explicit reactant/product compilation, analytic mass-action Jacobians,
  orthonormal even/odd mirror blocks, and a catalyst-preserving SynCRN adapter.
* Productive-but-stable and nonproductive-but-chirally-unstable controls,
  reverse catalytic channels, and racemization.
* Independently compiled archived C derivatives at 384 positive sample states
  and six selected finite-time trajectories.
* All physical racemic roots of an exact quartic, reference-rate threshold
  scans, and all 16 mirror/reverse module supports for each parent mechanism.
* Exhaustive small-support productive-core enumeration, paired nonlinear
  trajectories for every locally passing support, and comparisons against all
  proper module subsets at inherited rates.
* Selected full-parent racemization, local hydrolysis diagnostics, and the
  destructive-parent zero-reverse limit with positive reverse continuations.
* Machine-readable evidence, five paper figures, generated LaTeX numbers and
  tables, tighter-tolerance endpoint checks, and 11 regression tests of the
  scientific invariants.

Interpretation and boundaries
-----------------------------

Productivity allows internal rows L, D, LL, DD, LD, DL and excludes P, feed,
and washout. P remains dynamic in the ODE. Structural flux witnesses are not
claims about actual steady-state reaction fluxes. Exactly zero rates remove
channels from support; arbitrarily small positive rates retain them.

The four optional modules are matched, mismatched, homoligation, and
heteroligation. A deletion removes both mirrors and both reverse partners.
The environment is fixed for each 16-support comparison. Local chiral
instability and stable asymmetric endpoints are separate recorded checks.
Nonnegative attracting boundary endpoints are allowed. Inactive species stay
in the seven-species ODE with zero stationary concentrations.

Minimality is at two inherited parameter points, over the four-module
universe. It does not prove existential or robust minimality. Stable racemic
branches may coexist with distant asymmetric attractors or show transient
enrichment; the defined phenotype requires an odd instability from a small
seed near a racemate.

The reference C files are browser-text reconstructions, not raw-byte archive
downloads. ``results/source_manifest.json`` records the archive MD5 and actual
local SHA256/MD5. The entire archived main() sweep was not run. At positive
kdim the archived monomer hydrolysis convention differs away from LD=DL;
the difference is explicitly tested and described in the manuscript.

Autogatito and BiRNe were executed through ``backends.py`` on eight networks.
The separate Autogato repository was not run. LP enumeration remains the
baseline for the 32 supports. Shared conversion
equilibrium ratios and mirror symmetry are enforced; compatibility of the
experimental chemistry and a complete thermodynamic model remain open.

Main evidence files
-------------------

``results/summary.json`` summarizes scope and completed calculations.
``*_modules.json`` retains every support, parameter point, all racemic
states, spectra, residuals, and nonlinear endpoints. ``*_cores.json`` retains
productive-core supports and positive flux witnesses. ``*_trajectory.csv``
contains the paired time series. ``*_racemic_scan.csv``,
``*_racemization.csv``, and ``destructive_reverse_limit.csv`` contain plotted
data. ``results/provenance.json`` records script checksums and environment;
``results/source_manifest.json`` records external-source provenance.

Primary sources
---------------

Higgs--Blackmond paper: https://doi.org/10.1073/pnas.2423683122

Archived equations: https://zenodo.org/records/15299074

Explicit catalysis: https://arxiv.org/abs/2603.02770

BiRNe framework: https://arxiv.org/abs/2511.14431
