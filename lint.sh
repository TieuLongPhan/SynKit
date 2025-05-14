#!/usr/bin/env bash

flake8 . \
  --count \
  --max-complexity=13 \
  --max-line-length=120 \
  --per-file-ignores=\
"__init__.py:F401,\
chemical_reaction_visualizer.py:E501,\
test_reagent.py:E501,\
inference.py:F401,\
graph_morphism.py:C901,\
subgraph_matcher.py:C901,\
retro_reactor.py:C901,\
path_finder.py:C901,\
benchmarking_clustering.py:C901,\
benchmark_reactor.py:W292,C901,\
syn_reactor.py:C901,\
sing.py:C901,\
turbo_iso.py:C901,\
rule_vis.py:C901" \
  --exclude=venv,\
core_engine.py,\
rule_apply.py,\
reactor_engine.py,\
groupoid.py,\
syn_rule.py,\
__init__.py,\
Data \
  --statistics
