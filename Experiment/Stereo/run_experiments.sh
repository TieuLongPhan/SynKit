#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: Experiment/Stereo/run_experiments.sh TASK [ARGS...]

Tasks:
  canonicalization-local   Exhaustive fixed-graph local permutations
  canonicalization-global Selective whole-graph-by-local invariance
  chirality-published      Published ACS binary benchmark
  chirality-exact          Exact ACS configured-mirror audit
  chirality-relations      Designed stereoisomer-relation matrix
  perception-axis          RotA axial-locus benchmark
  perception-conformance   Designed task-aware conformance check
  perception-full          All ACS, RotA, and CIP carrier data (requires --cip-path)
  perception-cip           CIP local-label benchmark (requires --cip-path)
  perception-elements      Stereo-element inventory (requires --cip-path)
EOF
}

if [[ "$#" -lt 1 ]]; then
  usage
  exit 2
fi

task="$1"
shift

case "$task" in
  canonicalization-local)
    exec python Experiment/Stereo/Canonicalization/run.py "$@"
    ;;
  canonicalization-global)
    exec python Experiment/Stereo/Canonicalization/atom_relabel.py global-local "$@"
    ;;
  chirality-published)
    exec python Experiment/Stereo/Chirality/published.py "$@"
    ;;
  chirality-exact)
    exec python Experiment/Stereo/Chirality/run.py acs "$@"
    ;;
  chirality-relations)
    exec python Experiment/Stereo/Chirality/run.py relations "$@"
    ;;
  perception-axis)
    exec python Experiment/Stereo/Perception/axis_loci.py "$@"
    ;;
  perception-conformance)
    exec python Experiment/Stereo/Perception/conformance.py "$@"
    ;;
  perception-full)
    exec python Experiment/Stereo/Perception/full_detection.py "$@"
    ;;
  perception-cip)
    exec python Experiment/Stereo/Perception/cip_labels.py "$@"
    ;;
  perception-elements)
    exec python Experiment/Stereo/Perception/stereo_elements.py "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    printf 'Unknown stereo benchmark task: %s\n\n' "$task" >&2
    usage >&2
    exit 2
    ;;
esac
