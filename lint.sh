#!/bin/bash

flake8 . --count --max-complexity=13 --max-line-length=120 \
	--per-file-ignores="__init__.py:F401, chemical_reaction_visualizer.py:E501, test_reagent.py:E501, inference.py:F401, morphism.py:F401" \
	--exclude venv,core_engine.py,rule_apply.py \
	--statistics
