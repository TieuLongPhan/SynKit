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
[![Stars](https://img.shields.io/github/stars/tieulongphan/synkit.svg?style=social&label=Star)](https://github.com/tieulongphan/synkit/stargazers)

**Graph-native reaction informatics and executable electron-flow models**

SynKit is a Python toolkit for atom-mapped reactions, chemical graph
transformations, Lewis-labelled graphs, and explicitly supplied reaction
mechanisms. It connects reaction representation, rule extraction, graph
rewriting, stereochemistry, and mechanism verification through a common
attributed-graph model.

SynKit verifies mechanisms supplied by the user; it does not claim to predict
the kinetically or thermodynamically preferred mechanism.

![SynKit graphical abstract](https://raw.githubusercontent.com/TieuLongPhan/SynKit/main/Data/Figure/synkit_graphical_abstract.svg)

## Highlights

- Convert mapped reaction SMILES into ITS and reaction-centre graphs.
- Extract and apply attributed graph-transformation rules.
- Represent lone pairs, radicals, and separate sigma and pi occupancies with
  Lewis-labelled graphs.
- Execute curved-arrow and atomically coupled fishhook events.
- Preserve tetrahedral, axial, and extended stereochemical information.
- Build and analyse mechanism trajectory graphs and chemical reaction
  networks.

## Installation

SynKit requires Python 3.11 or later.

```bash
python -m pip install synkit
```

Install the optional dependencies with:

```bash
python -m pip install "synkit[all]"
```

Alternatively, use the published container:

```bash
docker pull tieulongphan/synkit:latest
docker run --rm tieulongphan/synkit:latest \
  python -c "import synkit; print(synkit.__version__)"
```

## Quick start

This self-contained example converts an atom-mapped substitution into an
imaginary transition-state graph and reports its changed bonds:

```python
from synkit.IO import rsmi_to_its

reaction = "[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3].[Br-:2]"
its = rsmi_to_its(reaction, core=False, format="tuple")

changed_bonds = [
    (source, target, data["order"])
    for source, target, data in its.edges(data=True)
    if data["order"][0] != data["order"][1]
]
print(changed_bonds)
```

The output records the broken C-Br bond as `(1.0, 0.0)` and the formed C-O
bond as `(0.0, 1.0)`. Continue with the
[graph](https://tieulongphan.github.io/SynKit/graph.html),
[rule](https://tieulongphan.github.io/SynKit/rule.html), and
[synthesis](https://tieulongphan.github.io/SynKit/synthesis.html) guides.

## Supplied-mechanism verification

SynKit can parse typed electron-flow annotations, replay their elementary
steps, and return a verification certificate and mechanism trajectory graph.
Canonical electron loci distinguish lone pairs, sigma bonds, pi bonds, and
radical electrons. Curved arrows carry two electrons; fishhooks carry one;
coupled radical events commit atomically from a common pre-state.

See the
[mechanism documentation](https://tieulongphan.github.io/SynKit/mechanism.html)
for the supported interchange formats, verification policies, and complete
examples.

## Documentation

- [Documentation](https://tieulongphan.github.io/SynKit/)
- [API reference](https://tieulongphan.github.io/SynKit/api/index.html)
- [Issue tracker](https://github.com/TieuLongPhan/SynKit/issues)
- [Changelog](doc/changelog.rst)

## Publications

- [Lewis-labeled graphs: curly arrows and fishhooks as executable electron
  transfers](https://arxiv.org/abs/2607.26088), submitted (2026).
- [SynKit: A Graph-Based Python Framework for Rule-Based Reaction Modeling and
  Analysis](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5c02123), *Journal
  of Chemical Information and Modeling* (2025).

## Contributing

Contributions and bug reports are welcome. Create a branch from the current
mainline, make a focused change, and run the repository checks before opening
a pull request:

```bash
git switch -c feature/short-description
bash scripts/lint.sh
bash scripts/pytest.sh
```

Questions can be sent to
[tieu@bioinf.uni-leipzig.de](mailto:tieu@bioinf.uni-leipzig.de).

## Contributors

- [Tieu-Long Phan](https://tieulongphan.github.io/)
- [Klaus Weinbauer](https://github.com/klausweinbauer)
- [Phuoc-Chung Nguyen Van](https://github.com/phuocchung123)
- [Tuyet-Minh Phan](https://github.com/tuyetminhphan)

## License

SynKit is distributed under the MIT License. See [LICENSE](LICENSE).

## Acknowledgments

This project received funding from the European Union's Horizon Europe
Doctoral Network programme under Marie Skłodowska-Curie grant agreement No.
101072930 ([TACsy](https://tacsy.eu/)).
