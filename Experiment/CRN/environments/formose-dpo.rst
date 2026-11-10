Formose DPO reproduction
=======================

The benchmark is shipped with the SynKit source distribution. It uses the four
mapped rules and two seed molecules defined in
``Experiment/CRN/studies/formose_case_study.py``. It does not import manuscript
scripts or archived result files. Native controller and typed DPO kernel builds
are explicit, content-addressed, and fingerprinted in the report.

Native prerequisites
--------------------

Install SynKit and its CRN environment constraints, a C++20 compiler, CMake >=
3.20, and an RDKit C++ SDK with Boost headers and CMake configuration. The native
controller uses RDKit for molecular IO and identity, and the SynKit DPO kernel
for graph transformations. The SDK need not share an address space with the
RDKit Python package. Set ``SYNKIT_RDKIT_CPP_DIR`` or pass ``--rdkit-dir``; no SDK
is downloaded automatically. The reference SDK was RDKit 2026.03.1.

From the unpacked source distribution, a short native check is::

    python Experiment/CRN/studies/formose_dpo.py \
      --rdkit-dir /path/to/rdkit-sdk --output-dir /tmp/formose-smoke \
      --depths 1 2 3 4 --warmups 1 --repeats 2

The complete native measurement workload is::

    python Experiment/CRN/studies/formose_dpo.py \
      --rdkit-dir /path/to/rdkit-sdk --output-dir /tmp/formose-native \
      --depths 1 2 3 4 5 6 --warmups 3 --repeats 20

Independent engine comparison
-----------------------------

Create a MØD environment using ``mod-environment.yml`` and a separate Python
virtual environment using ``molfoundry-requirements.txt``. The pins fix the engine
versions used for the comparison; they are not full operating-system lockfiles.
MolFoundry is distributed through ``https://py.molfoundry.de/dl/``; its custom
package index is included in the requirements file.
The benchmark records the engine and Python versions actually executed. Neither
external environment needs a SynKit installation or RDKit Python package;
format translation and network canonicalization run in the main interpreter.

Require all three engines with::

    python Experiment/CRN/studies/formose_dpo.py \
      --rdkit-dir /path/to/rdkit-sdk --output-dir /tmp/formose-three-tools \
      --external --mod-python /path/to/mod-env/bin/python \
      --molfoundry-python /path/to/molfoundry-env/bin/python \
      --depths 1 2 3 4 5 6 --warmups 3 --repeats 20

A missing requested engine, timeout, incomplete native expansion, unstable
repetition, mismatching network, or source modification makes the report fail
and the command return nonzero. Native-only runs explicitly set
``cross_tool_equality_checked`` to false and cannot establish cross-tool
agreement. The report writes one row for every requested engine and depth.

Timing and verification
-----------------------

Each timed call prepares fresh seeds and rules and constructs the network through
the tool's public API. Imports, native compilation, controller startup, fixed GML
translation, graph export, and the independent molecular identity audit are
outside timing. Warmups are distinct from measured samples. Only one expansion
is active at once, with tool order rotated across repetitions and depths.

On POSIX the default per-worker address-space bound is 8 GiB; Linux workers are
pinned to one available CPU by default. The default timeout is 180 seconds per
expansion, with separate startup and audit budgets. Limits and whether they are
enforced are recorded. CPU affinity and address-space controls may differ on
other platforms; benchmark times are machine-specific and must be regenerated.

Canonicalization removes explicit hydrogen vertices and stereochemistry and
compares full species sets and directed reactant/product multisets. Shared
participants are retained, and stoichiometric multiplicities are counted. Every
repetition must have the same snapshot hash. The report contains the exact rule
inputs, source hashes, controller build manifest, all samples, full execution
schedule, and separate snapshots for inspection. Completed depths 1--6 have
reference species/reaction counts of 3/1, 5/4, 9/10, 37/44, 302/371, and
10572/11239. These checks do not assert kinetic or thermodynamic equivalence.
