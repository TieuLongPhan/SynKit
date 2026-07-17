# SynKit
[![PyPI version](https://img.shields.io/pypi/v/synkit.svg)](https://pypi.org/project/synkit/)
[![Conda version](https://img.shields.io/conda/vn/tieulongphan/synkit.svg)](https://anaconda.org/tieulongphan/synkit)
[![Docker Pulls](https://img.shields.io/docker/pulls/tieulongphan/synkit.svg)](https://hub.docker.com/r/tieulongphan/synkit)
[![Docker Image Version](https://img.shields.io/docker/v/tieulongphan/synkit/latest?label=container)](https://hub.docker.com/r/tieulongphan/synkit)
[![License](https://img.shields.io/github/license/tieulongphan/synkit.svg)](https://github.com/tieulongphan/synkit/blob/main/LICENSE)
[![Release](https://img.shields.io/github/v/release/tieulongphan/synkit.svg)](https://github.com/tieulongphan/synkit/releases)
[![Last Commit](https://img.shields.io/github/last-commit/tieulongphan/synkit.svg)](https://github.com/tieulongphan/synkit/commits)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.15269901.svg)](https://doi.org/10.5281/zenodo.15269901)
[![CI](https://github.com/tieulongphan/synkit/actions/workflows/test-and-lint.yml/badge.svg?branch=main)](https://github.com/tieulongphan/synkit/actions/workflows/test-and-lint.yml)
[![Dependency PRs](https://img.shields.io/github/issues-pr-raw/tieulongphan/synkit?label=dependency%20PRs)](https://github.com/tieulongphan/synkit/pulls?q=is%3Apr+label%3Adependencies)
[![Stars](https://img.shields.io/github/stars/tieulongphan/synkit.svg?style=social&label=Star)](https://github.com/tieulongphan/synkit/stargazers)

**Graph-native reaction informatics and supplied-mechanism verification**

SynKit represents mapped reactions, Lewis-state graphs, transformation rules,
and explicitly supplied electron-flow mechanisms. The in-development v2 API
adds atomic curved-arrow/fishhook groups, radical state, relative atom- and
bond-centered stereochemistry, deterministic replay certificates, and
mechanism trajectory graphs. It validates annotations; it does not predict the
chemically preferred mechanism.

![SynKit](https://raw.githubusercontent.com/TieuLongPhan/SynKit/main/Data/Figure/synkit.png)

### Mechanism verification quick start

```python
from synkit.Mechanism import MechanismRecord

mechanism = MechanismRecord.from_ef_smirks(text)
certificate = mechanism.verify(electron="strict", stereo="stepwise")
trajectory = mechanism.to_mtg()
mechanism.draw(certificate=certificate, path="mechanism.svg")
mechanism.to_json("mechanism.json")
```

Canonical internal electron loci are `lp`, `σ`, `π`, and `∙`; adapters accept
documented ASCII and legacy spellings. Curved arrows carry two electrons and
fishhooks carry one electron. Coupled radical events commit atomically.

SynKit graph/rule metadata supports relative tetrahedral, square-planar,
trigonal-bipyramidal, octahedral, planar double-bond, and atrop-bond stereo.
The RDKit adapter round-trips all six families when the state is representable:
tetrahedral and planar-bond stereo plus assigned square-planar,
trigonal-bipyramidal, octahedral, and atrop-bond descriptors. Unknown
non-tetrahedral or atrop orientation fails explicitly instead of disappearing.
Coordinate inference, rigid-bond descriptor variants, enhanced stereo groups,
face-selectivity prediction, and physical inference of configurational
stability remain outside the current claims. The prerelease package version is
`2.0.0b1`; its mechanism schema remains `2.0.0-draft1` while that schema is
still under prerelease review.

Stereo-bearing reaction rules keep three concerns separate:
`stereo_guards` constrain reactant mappings, `stereo_effects` store stable
before/after descriptors plus optional transition-state stereo, and
`stereo_outcomes` control product distributions. `RACEMIC` always means an
equal 0.5/0.5 pair; unequal weights use `ENANTIOMERIC_MIXTURE`. Unknown parity
is exact unless a rule or reactor explicitly selects wildcard query semantics.
Reversing a branching rule accepts either reactant enantiomer and emits one
reverse product instead of incorrectly branching a second time.

`stereo_isomorphic(left, right)` enumerates structural graph isomorphisms and
accepts only mappings that preserve the complete relative descriptor registry.
This prevents a connectivity-only mapping from equating enantiomers while
still allowing a later stereo-valid mapping in a symmetric graph.

For more details on each utility within the repository, please refer to the documentation provided in the respective folders.

## Table of Contents
- [Installation](#installation)
- [Contribute to `SynKit`](#contribute)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

1. **Python Installation:**
  Ensure that Python 3.11 or later is installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

2. **Creating a Virtual Environment (Optional but Recommended):**
  It's recommended to use a virtual environment to avoid conflicts with other projects or system-wide packages. Use the following commands to create and activate a virtual environment:

  ```bash
  python -m venv synkit-env
  source synkit-env/bin/activate  
  ```
  Or Conda

  ```bash
  conda create --name synkit-env python=3.11
  conda activate synkit-env
  ```

3. **Install from PyPi:**
  The easiest way to use SynTemp is by installing the PyPI package 
  [synkit](https://pypi.org/project/synkit/).

  ```
  pip install synkit
  ```
  Optional if you want to install full version
  ```
  pip install synkit[all]
  ```

4. **Install via Docker**  
   Pull the image: 

   ```bash
   docker pull tieulongphan/synkit:latest
   # or a specific version:
   docker pull tieulongphan/synkit:1.0.0
   ```
   Run a container (sanity check):
   ```
   docker run --rm tieulongphan/synkit:latest
   ```

## Contribute

We're welcoming new contributors to build this project better. Please not hesitate to inquire me via [email](tieu@bioinf.uni-leipzig.de).

Before you start, ensure your local development environment is set up correctly. Pull the latest version of the `main` branch to start with the most recent stable code.

```bash
git checkout main
git pull
```

### Working on New Features

1. **Create a New Branch**:  
   For every new feature or bug fix, create a new branch from the `main` branch. Name your branch meaningfully, related to the feature or fix you are working on.

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Develop and Commit Changes**:  
   Make your changes locally, commit them to your branch. Keep your commits small and focused; each should represent a logical unit of work.

   ```bash
   git commit -m "Describe the change"
   ```

3. **Run Quality Checks**:  
   Before finalizing your feature, run the following commands to ensure your code meets our formatting standards and passes all tests:

   ```bash
   ./lint.sh # Check code format
   pytest Test # Run tests
   ```

   Fix any issues or errors highlighted by these checks.

### Integrating Changes

1. **Rebase onto Staging**:  
   Once your feature is complete and tests pass, rebase your changes onto the `staging` branch to prepare for integration.

   ```bash
   git fetch origin
   git rebase origin/staging
   ```

   Carefully resolve any conflicts that arise during the rebase.

2. **Push to Your Feature Branch**:
   After successfully rebasing, push your branch to the remote repository.

   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request**:
   Open a pull request from your feature branch to the `staging` branch. Ensure the pull request description clearly describes the changes and any additional context necessary for review.

## Contributing
- [Tieu-Long Phan](https://tieulongphan.github.io/)
- [Klaus Weinbauer](https://github.com/klausweinbauer)
- [Phuoc-Chung Nguyen Van](https://github.com/phuocchung123)
- [Tuyet-Minh Phan](https://github.com/tuyetminhphan)

## Publication

[**SynKit**: A Graph-Based Python Framework for Rule-Based Reaction Modeling and Analysis](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5c02123)


## License

This project is licensed under MIT License - see the [License](LICENSE) file for details.

## Acknowledgments

The relative-stereo rule model is implemented from the ideas in M.
Papusha and K. Leonhard, “StereoMolGraph: Stereochemistry-Aware Molecular and
Reaction Graphs,” *J. Chem. Inf. Model.* (2026), and the MIT-licensed
[StereoMolGraph repository](https://github.com/maxim-papusha/StereoMolGraph),
cross-validation commit `2189f610f23eaaf992e2e01a12ea4d0532496601`.
Non-tetrahedral permutation groups are adapted under the upstream MIT license;
the retained notice is in
[`LICENSES/StereoMolGraph-MIT.txt`](LICENSES/StereoMolGraph-MIT.txt).
The optional development check is run with
`python tools/stereo_conformance.py /path/to/StereoMolGraph`; StereoMolGraph is
not a SynKit runtime dependency. Its unknown-parity wildcard equality is an
intentional difference: SynKit keeps stored-value equality exact and exposes
wildcards only through explicit rule-query policy.

This project has received funding from the European Unions Horizon Europe Doctoral Network programme under the Marie-Skłodowska-Curie grant agreement No 101072930 ([TACsy](https://tacsy.eu/) -- Training Alliance for Computational)
