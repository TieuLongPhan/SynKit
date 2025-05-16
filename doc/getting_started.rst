.. _getting-started-synkit:

Getting Started
===============

Welcome to the **synkit** documentation! This guide walks you through installing and verifying **synkit**, a modular toolkit for chemical informatics and graph-based workflows.

Introduction
------------
**synkit** provides a suite of tools to simplify reaction canonicalization, atom-map validation, graph transformation, and more. Whether you’re automating chemical data pipelines or building custom rule-based systems, **synkit** helps you get started quickly and scale confidently.

Requirements
------------
Before installing **synkit**, ensure that:

- **Python** ≥ 3.11 is available on your system.  
- You have a working C/C++ compiler for any native extensions.  
- (Recommended) You use an isolated virtual environment to avoid dependency conflicts.

Virtual Environment (Recommended)
---------------------------------
Creating an isolated environment prevents conflicts between **synkit** and other Python projects.

1. **Using venv** (cross-platform)

   .. code-block:: bash

      python3 -m venv synkit-env
      source synkit-env/bin/activate    # Linux/macOS
      synkit-env\Scripts\activate       # Windows PowerShell

2. **Using Conda** (if you prefer Conda environments)

   .. code-block:: bash

      conda create -n synkit-env python=3.11
      conda activate synkit-env

Installing Dependencies
-----------------------
Some **synkit** features require the external package **mod**.

.. note::
   On Linux you can install **mod** via Conda:

   .. code-block:: bash

      conda install -c jakobandersen -c conda-forge "mod>=0.17" -y

   For other platforms, see the upstream instructions:  
   <https://jakobandersen.github.io/mod/installation.html>_

Installing synkit
-----------------
With your environment activated and dependencies in place, install **synkit** from PyPI:

.. code-block:: bash

   pip install synkit

This will pull in **synkit** and all required dependencies.

Quick Verification
------------------
After installation, verify that **synkit** is available and check its version:

.. code-block:: bash

   python -c "import synkit; print(synkit.__version__)"
   # Should print the installed synkit version

Further Resources
-----------------
- Official documentation: `SynKit Docs <https://tieulongphan.github.io/SynKit>`_
- Tutorials and examples: :doc:`Tutorials and Examples <getting_started>`

Support
-------
If you encounter issues or have questions:

- Report bugs and feature requests on GitHub:  
  `SynKit Issues <https://github.com/TieuLongPhan/SynKit/issues>`_

Enjoy using **synkit**!
