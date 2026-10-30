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

Requires Python 3.10 or later.

```bash
python -m pip install synkit
```

See [Getting Started](doc/getting_started.rst) for examples, optional
dependencies, and Docker installation.

Platform wheels include the C++ PABS–SLAP atom mapper. It searches heavy-atom
classes with hydrogen changes included in the Lewis score:

```python
from synkit.Chem.Mapper import map_reaction_implicit

result = map_reaction_implicit("CCO>>COC", time_limit_seconds=10)
print(result.minimum_cost, result.complete)
print(result.mapped_reactions)
```

See [Atom mapping](doc/mapper.rst) for exact-CD enumeration, solver selection
and custom builds. Older releases may require an explicit native build.

The same wheel also bundles exact CRN analysis and the graph canonicalization
backend used by native stereograph analysis. Select native CRN analysis with:

```python
from synkit.CRN import SynCRN, crnt_summary, native_backend

model = SynCRN.from_reaction_strings(["A>>B", "B>>A"])
with native_backend():
    print(crnt_summary(model).to_dict())
```

Platform wheels need no local compiler or Boost installation. Source builders
can compile all native components together using the
[shared build instructions](scripts/README.md#native-builds).

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

Questions can be sent to [tieu@bioinf.uni-leipzig.de](mailto:tieu@bioinf.uni-leipzig.de).

## Contributors

- [Tieu-Long Phan](https://tieulongphan.github.io/)
- [Klaus Weinbauer](https://github.com/klausweinbauer)
- [Phuoc-Chung Nguyen Van](https://github.com/phuocchung123)
- [Tuyet-Minh Phan](https://github.com/tuyetminhphan)

## License

SynKit is distributed under the MIT License. See [LICENSE](LICENSE).

## Acknowledgments

This project received funding from the European Union’s Horizon Europe
Doctoral Network programme under Marie Skłodowska-Curie grant agreement
No. 101072930 ([TACsy](https://tacsy.eu/)).
