# SynKit

**Toolkit for Synthesis Planning**

SynKit is a collection of tools designed to support the planning and execution of chemical synthesis. 

![SynKit](https://raw.githubusercontent.com/TieuLongPhan/SynKit/main/Data/Figure/synkit.png)

Our tools are tailored to assist researchers and chemists in navigating complex chemical reactions and synthesis pathways, leveraging the power of modern computational chemistry. Whether you're designing novel compounds or optimizing existing processes, ``synkit`` aims to provide the critical tools you need.

For more details on each utility within the repository, please refer to the documentation provided in the respective folders.

## Step-by-Step Installation Guide

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

## For contributors

We're welcoming new contributors to build this project better. Please not hesitate to inquire me via [email][tieu@bioinf.uni-leipzig.de].

Before you start, ensure your local development environment is set up correctly. Pull the latest version of the `main` branch to start with the most recent stable code.

```bash
git checkout main
git pull
```

## Working on New Features

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

## Integrating Changes

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
   Open a pull request from your feature branch to the `stagging` branch. Ensure the pull request description clearly describes the changes and any additional context necessary for review.

## Contributing
- [Tieu-Long Phan](https://tieulongphan.github.io/)
- [Klaus Weinbauer](https://github.com/klausweinbauer)
- [Phuoc-Chung Nguyen Van](https://github.com/phuocchung123)

## Deployment timeline

We plan to update new version quarterly.


## License

This project is licensed under MIT License - see the [License](LICENSE) file for details.

## Acknowledgments

This project has received funding from the European Unions Horizon Europe Doctoral Network programme under the Marie-Skłodowska-Curie grant agreement No 101072930 ([TACsy](https://tacsy.eu/) -- Training Alliance for Computational)