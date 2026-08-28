#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 DATASET [OUTPUT]" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
if [[ ! -f "$1" ]]; then
    echo "dataset does not exist: $1" >&2
    exit 2
fi
dataset="$(realpath -- "$1")"
output_input="${2:-benchmark_results/synister_global_shells_v4_60s_w16}"
if [[ "${output_input}" = /* ]]; then
    output_candidate="${output_input}"
else
    output_candidate="${repo_root}/${output_input}"
fi
mkdir -p -- "${output_candidate}"
output="$(realpath -- "${output_candidate}")"
python_bin="$(command -v python)"
unit="synister-global-shells-v4-60s-w16"
log_path="${output}/campaign.log"

systemctl --user reset-failed "${unit}.service" >/dev/null 2>&1 || true
systemd-run --user --collect \
    --unit="${unit}" \
    --description="Synister global shells: 16 workers, 60-second shells" \
    --working-directory="${repo_root}" \
    --property=CPUQuota=1600% \
    --property=CPUWeight=200 \
    --property=MemoryHigh=8G \
    --property=MemoryMax=10G \
    --property=MemorySwapMax=0 \
    --property=OOMPolicy=stop \
    --property="StandardOutput=append:${log_path}" \
    --property="StandardError=append:${log_path}" \
    "${python_bin}" scripts/run_synister_global_shells.py \
    --dataset "${dataset}" \
    --output "${output}" \
    --mode both \
    --workers 16 \
    --time-limit-per-shell 60 \
    --memory-limit-gib 4

echo "service: ${unit}.service"
echo "log: ${log_path}"
echo "status: systemctl --user status ${unit}.service"
