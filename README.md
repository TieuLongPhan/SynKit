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

**Toolkit for Synthesis Planning**

SynKit is a collection of tools designed to support the planning and execution of chemical synthesis. Check out the [documentation](https://tieulongphan.github.io/SynKit/) for a comprehensive description of its features.

![SynKit](https://raw.githubusercontent.com/TieuLongPhan/SynKit/main/Data/Figure/synkit.png)

Our tools are tailored to assist researchers and chemists in navigating complex chemical reactions and synthesis pathways, leveraging the power of modern computational chemistry. Whether you're designing novel compounds or optimizing existing processes, ``synkit`` aims to provide the critical tools you need.

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

This project has received funding from the European Unions Horizon Europe Doctoral Network programme under the Marie-Skłodowska-Curie grant agreement No 101072930 ([TACsy](https://tacsy.eu/) -- Training Alliance for Computational)