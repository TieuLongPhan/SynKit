#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
cd "${ROOT_DIR}"
if [[ "$#" -gt 0 ]]; then
  paths=("$@")
else
  paths=(synkit Test)
fi
"${PYTHON_BIN}" scripts/check_python_file_size.py
"${PYTHON_BIN}" scripts/check_docstring_style.py
flake8 "${paths[@]}" \
  --count \
  --max-complexity=13 \
  --max-line-length=120 \
  --extend-ignore=E203 \
  --per-file-ignores="\
__init__.py:F401,F403,\
synkit/Chem/Reaction/explicit_h_audit.py:C901,\
synkit/Chem/Reaction/Mapper/wl_mapper.py:C901,\
synkit/Graph/FG/detector.py:C901,\
synkit/Graph/Feature/Descriptors/topology.py:C901,E501,\
synkit/Graph/ITS/its_destruction.py:C901,\
synkit/Graph/Stereo/rdkit_adapter.py:C901,\
synkit/Graph/Matcher/graph_morphism.py:C901,\
synkit/Graph/Matcher/sing.py:C901,\
synkit/Graph/Matcher/subgraph_matcher.py:C901,\
synkit/Graph/Matcher/turbo_iso.py:C901,\
synkit/Graph/MTG/mtg.py:C901,\
synkit/Graph/Wildcard/wildcard.py:C901,\
synkit/Graph/Wildcard/injectivity.py:C901,\
synkit/Graph/Wildcard/its_merge.py:C901,\
synkit/IO/chem_converter.py:C901,\
synkit/IO/combinatorial/gml_to_graph.py:C901,\
synkit/IO/gml_to_graph.py:C901,\
synkit/IO/mol_to_graph.py:C901,\
synkit/IO/mol_to_graph_chemistry.py:C901,\
synkit/Mechanism/stereo_state.py:C901,\
synkit/Rule/syn_rule.py:C901,\
synkit/Rule/Apply/retro_reactor.py:C901,\
synkit/Rule/Apply/syn_reactor.py:C901,\
synkit/Rule/Apply/rule_apply.py:C901,\
synkit/Synthesis/MSR/path_finder.py:C901,\
synkit/Synthesis/RBL/engine.py:C901,\
synkit/Synthesis/Reactor/core/rewrite.py:C901,\
synkit/Synthesis/Reactor/core/engine.py:C901,\
synkit/Synthesis/Reactor/matching/mixin.py:C901,\
synkit/Synthesis/Reactor/output/serialization.py:C901,\
synkit/Synthesis/Reactor/stereo/mixin.py:C901,\
synkit/Vis/reaction/rule.py:C901" \
  --exclude="\
venv,\
.venv,\
__pycache__,\
.git,\
.pytest_cache,\
Data,\
debug/data,\
dev,\
docs,\
doc,\
synkit/Graph/dev" \
  --statistics
