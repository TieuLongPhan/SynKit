#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_bidirectional_replay.sh [OUTPUT_DIR]

Run all ten compressed FLOWER full-reaction batches in both directions.

Environment:
  PYTHON                Python executable with SynKit and RDKit (default: python)
  JOBS                  Concurrent batches / approximate CPU cores (default: 1)
  CASE_TIMEOUT          Per-direction seconds; use 0 for no timeout (default: 30)
  FAILURE_SAMPLE_LIMIT  Generated failure examples to retain (default: 0)
  PROGRESS_EVERY        Rows between progress flushes (default: 250)
  VERIFY_BATCHES        Verify manifest SHA-256 values first: 1 or 0 (default: 1)

Examples:
  bash Experiment/Lewis/FLOWER/run_bidirectional_replay.sh
  JOBS=8 PYTHON=/opt/conda/envs/synkit/bin/python \
    bash Experiment/Lewis/FLOWER/run_bidirectional_replay.sh /data/flower-replay
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi
if (( $# > 1 )); then
    usage >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
batch_dir="${script_dir}/full-reaction-batches"
manifest="${batch_dir}/manifest.json"
replay_py="${script_dir}/replay.py"

python_bin="${PYTHON:-python}"
jobs="${JOBS:-1}"
case_timeout="${CASE_TIMEOUT:-30}"
failure_sample_limit="${FAILURE_SAMPLE_LIMIT:-0}"
progress_every="${PROGRESS_EVERY:-250}"
verify_batches="${VERIFY_BATCHES:-1}"
output_dir="${1:-${script_dir}/replay-results}"

if [[ ! ${jobs} =~ ^[1-9][0-9]*$ ]] || (( jobs > 10 )); then
    echo "JOBS must be an integer from 1 to 10" >&2
    exit 2
fi
if [[ ! ${failure_sample_limit} =~ ^[0-9]+$ ]]; then
    echo "FAILURE_SAMPLE_LIMIT must be a non-negative integer" >&2
    exit 2
fi
if [[ ! ${progress_every} =~ ^[0-9]+$ ]]; then
    echo "PROGRESS_EVERY must be a non-negative integer" >&2
    exit 2
fi
if [[ ${verify_batches} != "0" && ${verify_batches} != "1" ]]; then
    echo "VERIFY_BATCHES must be 0 or 1" >&2
    exit 2
fi
if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "Python executable not found: ${python_bin}" >&2
    exit 1
fi
if [[ ! -f ${manifest} || ! -f ${replay_py} ]]; then
    echo "Run this script from a complete SynKit checkout with FLOWER batches" >&2
    exit 1
fi

shopt -s nullglob
batches=("${batch_dir}"/batch-??-of-10.txt.gz)
shopt -u nullglob
if (( ${#batches[@]} != 10 )); then
    echo "Expected 10 compressed batches in ${batch_dir}; found ${#batches[@]}" >&2
    exit 1
fi

if [[ -d ${output_dir} ]] &&
    [[ -n "$(find "${output_dir}" -mindepth 1 -print -quit)" ]]; then
    echo "Output directory is not empty: ${output_dir}" >&2
    echo "Choose a new directory so existing replay evidence is not overwritten." >&2
    exit 1
fi
mkdir -p "${output_dir}"
output_dir="$(cd -- "${output_dir}" && pwd)"

cd "${repo_root}"
"${python_bin}" -c "import rdkit, synkit" >/dev/null

if [[ ${verify_batches} == "1" ]]; then
    "${python_bin}" - "${manifest}" "${batch_dir}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1])
batch_dir = Path(sys.argv[2])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
entries = payload.get("batches", [])
if len(entries) != 10:
    raise SystemExit(f"Manifest contains {len(entries)} batches, expected 10")
for entry in entries:
    path = batch_dir / Path(entry["path"]).name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise SystemExit(f"SHA-256 mismatch: {path}")
print(
    f"Verified {len(entries)} batches: "
    f"{payload['total_rows']:,} rows / "
    f"{payload['total_directional_replays']:,} directions"
)
PY
fi

timeout_args=()
if [[ ${case_timeout} != "0" ]]; then
    timeout_args=(--case-timeout "${case_timeout}")
fi

cat >"${output_dir}/run-config.txt" <<EOF
python=${python_bin}
jobs=${jobs}
case_timeout=${case_timeout}
failure_sample_limit=${failure_sample_limit}
progress_every=${progress_every}
manifest=${manifest}
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

run_batch() {
    local batch="$1"
    local name="${batch##*/}"
    local tag="${name#batch-}"
    tag="${tag%%-of-*}"
    local batch_output="${output_dir}/batch-${tag}"

    mkdir -p "${batch_output}"
    echo "Starting batch ${tag}: ${name}"
    "${python_bin}" "${replay_py}" "${batch}" \
        "${timeout_args[@]}" \
        --failure-sample-limit "${failure_sample_limit}" \
        --progress-every "${progress_every}" \
        --output-dir "${batch_output}" \
        2>&1 | tee "${batch_output}/runner.log"
}

declare -a active_pids=()
declare -a active_names=()
run_status=0

wait_for_first() {
    local pid="${active_pids[0]}"
    local name="${active_names[0]}"
    if wait "${pid}"; then
        echo "Completed ${name}"
    else
        echo "Failed ${name}" >&2
        run_status=1
    fi
    active_pids=("${active_pids[@]:1}")
    active_names=("${active_names[@]:1}")
}

for batch in "${batches[@]}"; do
    run_batch "${batch}" &
    active_pids+=("$!")
    active_names+=("${batch##*/}")
    if (( ${#active_pids[@]} >= jobs )); then
        wait_for_first
    fi
done
while (( ${#active_pids[@]} )); do
    wait_for_first
done

aggregate_bugs="${output_dir}/bugs.jsonl"
: >"${aggregate_bugs}"
for batch in "${batches[@]}"; do
    name="${batch##*/}"
    tag="${name#batch-}"
    tag="${tag%%-of-*}"
    batch_bugs="${output_dir}/batch-${tag}/bugs.jsonl"
    if [[ -f ${batch_bugs} ]]; then
        cat "${batch_bugs}" >>"${aggregate_bugs}"
    fi
done

bug_count=$(wc -l <"${aggregate_bugs}")
{
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "aggregate_bug_rows=${bug_count}"
    echo "exit_status=${run_status}"
} >>"${output_dir}/run-config.txt"

echo "Aggregate bug log: ${aggregate_bugs} (${bug_count} rows)"
echo "Replay outputs: ${output_dir}"
exit "${run_status}"
