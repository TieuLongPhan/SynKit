# SynKit

[![PyPI version](https://img.shields.io/pypi/v/synkit.svg)](https://pypi.org/project/synkit/)
[![License](https://img.shields.io/github/license/tieulongphan/synkit.svg)](https://github.com/tieulongphan/synkit/blob/main/LICENSE)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.15269901.svg)](https://doi.org/10.5281/zenodo.15269901)
[![CI](https://github.com/tieulongphan/synkit/actions/workflows/test-and-lint.yml/badge.svg?branch=main)](https://github.com/tieulongphan/synkit/actions/workflows/test-and-lint.yml)

**Graph-native reaction informatics and executable electron-flow models**

SynKit is a Python toolkit for atom mapping, chemical graph transformations,
reaction rules, stereochemistry, and chemical reaction networks. It also
represents Lewis-labelled graphs and verifies user-supplied electron-flow
mechanisms. It does not predict the kinetically or thermodynamically preferred
mechanism.

![SynKit graphical abstract](https://raw.githubusercontent.com/TieuLongPhan/SynKit/main/Data/Figure/synkit_graphical_abstract.svg)

## Installation

Requires Python 3.11 or later.

```bash
python -m pip install synkit
```

See [Getting Started](doc/getting_started.rst) for examples, optional
dependencies, and Docker installation.

## Documentation

- [Getting Started](doc/getting_started.rst)
- [Chemistry and atom mapping](doc/chem.rst#atom-to-atom-mapping)
- [Graphs](doc/graph.rst), [reaction rules](doc/rule.rst), and
  [synthesis](doc/synthesis.rst)
- [Chemical reaction networks](doc/crn.rst)
- [Documentation website](https://tieulongphan.github.io/SynKit/) and
  [API reference](https://tieulongphan.github.io/SynKit/api/index.html)
- [Changelog](doc/changelog.rst)

## Publications

- [SynKit: A Graph-Based Python Framework for Rule-Based Reaction Modeling and
  Analysis](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5c02123), *Journal
  of Chemical Information and Modeling* (2025).
- [Lewis-labeled graphs: curly arrows and fishhooks as executable electron
  transfers](https://arxiv.org/abs/2607.26088), submitted (2026).

## Contributing

Open an [issue](https://github.com/TieuLongPhan/SynKit/issues) or a pull request.
Before submitting code, run `bash scripts/lint.sh` and `bash scripts/pytest.sh`.
Contact: [tieu@bioinf.uni-leipzig.de](mailto:tieu@bioinf.uni-leipzig.de).

## Contributors

- [Tieu-Long Phan](https://tieulongphan.github.io/)
- [Klaus Weinbauer](https://github.com/klausweinbauer)
- [Phuoc-Chung Nguyen Van](https://github.com/phuocchung123)
- [Tuyet-Minh Phan](https://github.com/tuyetminhphan)

## License and funding

Distributed under the MIT License; see [LICENSE](LICENSE).

Funded by the European Union's Horizon Europe Doctoral Network programme,
Marie Skłodowska-Curie grant No. 101072930 ([TACsy](https://tacsy.eu/)).
