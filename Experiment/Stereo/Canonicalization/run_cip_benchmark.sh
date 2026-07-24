#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

MODE="${1:-full}"
CIP_FILE="${SYNKIT_CIP_FILE:-/tmp/cip-validation-suite-compounds.smi}"
JOBS="${SYNKIT_CIP_JOBS:-4}"
TIMEOUT="${SYNKIT_CIP_TIMEOUT:-10}"

CIP_REVISION="6b9f9db46dadc6749da8234b05164e1e0fb413b9"
CIP_SHA256="df178635c00b6c41fad820d2609fc4ff18403c63dec4e5c3e1756a6db5858059"
CIP_URL="https://raw.githubusercontent.com/CIPValidationSuite/ValidationSuite/${CIP_REVISION}/compounds.smi"

case "${MODE}" in
    full | ct4)
        ;;
    *)
        echo "Usage: $0 [full|ct4]" >&2
        exit 2
        ;;
esac

if [[ ! -f "${CIP_FILE}" ]] || \
    [[ "$(sha256sum "${CIP_FILE}" | cut -d' ' -f1)" != "${CIP_SHA256}" ]]; then
    echo "Downloading the pinned CIP Validation Suite..."
    curl -fL "${CIP_URL}" -o "${CIP_FILE}"
fi

echo "${CIP_SHA256}  ${CIP_FILE}" | sha256sum -c -

cd "${REPOSITORY_ROOT}"

COMMAND=(
    env PYTHONPATH=.
    conda run -n synkit python
    Experiment/Stereo/Canonicalization/run.py
    cip
    --cip-path "${CIP_FILE}"
    --timeout "${TIMEOUT}"
    --jobs "${JOBS}"
)

if [[ "${MODE}" == "ct4" ]]; then
    COMMAND+=(
        --record-id VS063
        --record-id VS118
        --record-id VS135
        --record-id VS154
        --record-id VS164
        --output /tmp/synkit-ct4-cip-report.json
        --table /tmp/synkit-ct4-cip-table.csv
    )
fi

echo "Running ${MODE} CIP benchmark with ${JOBS} workers..."
"${COMMAND[@]}"
