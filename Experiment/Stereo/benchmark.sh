#!/usr/bin/env bash
set -euo pipefail

# Run every maintained Stereo benchmark and save the reports locally.
# The external CIP structure file is intentionally kept outside the repository.
# Case-based tasks use up to 16 workers by default. Each worker's numerical
# libraries remain single-threaded to prevent oversubscription.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPOSITORY_ROOT}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
BENCHMARK_JOBS="${SYNKIT_BENCHMARK_JOBS:-16}"
if ! [[ "${BENCHMARK_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SYNKIT_BENCHMARK_JOBS must be a positive integer." >&2
  exit 2
fi

CANON_LOCAL="${REPOSITORY_ROOT}/Experiment/Stereo/Data/Canonicalization"
CANON_GLOBAL="${REPOSITORY_ROOT}/Experiment/Stereo/Data/Canonicalization"
CANON_MULTI="${REPOSITORY_ROOT}/Experiment/Stereo/Data/Canonicalization"
CANON_ABC="${REPOSITORY_ROOT}/Experiment/Stereo/Data/Canonicalization/Global"
CHIRALITY_PUBLISHED="${REPOSITORY_ROOT}/Experiment/Stereo/Data/Chirality"
CHIRALITY_EXACT="${REPOSITORY_ROOT}/Experiment/Stereo/Data/Chirality"
CHIRALITY_RELATIONS="${REPOSITORY_ROOT}/Experiment/Stereo/Data/Chirality"
PERCEPTION_AXIS="${REPOSITORY_ROOT}/Experiment/Stereo/Data/Perception"
PERCEPTION_CONFORMANCE="${REPOSITORY_ROOT}/Experiment/Stereo/Data/Perception"
PERCEPTION_FULL="${REPOSITORY_ROOT}/Experiment/Stereo/Data/Perception"
PERCEPTION_CIP="${REPOSITORY_ROOT}/Experiment/Stereo/Data/Perception"
PERCEPTION_ELEMENTS="${REPOSITORY_ROOT}/Experiment/Stereo/Data/Perception"
PERCEPTION_ROTA_NEGATIVE="${REPOSITORY_ROOT}/Experiment/Stereo/Data/RotA-Synthetic"
PERCEPTION_CIP_CONTRACT="${REPOSITORY_ROOT}/Experiment/Stereo/Data/CIP"
DIAGNOSTICS="${REPOSITORY_ROOT}/Experiment/Stereo/Data/Diagnostics"

mkdir -p \
  "${CANON_LOCAL}" "${CANON_GLOBAL}" "${CANON_MULTI}" \
  "${CANON_ABC}/A" "${CANON_ABC}/B" "${CANON_ABC}/C" \
  "${CHIRALITY_PUBLISHED}" "${CHIRALITY_EXACT}" "${CHIRALITY_RELATIONS}" \
  "${PERCEPTION_AXIS}" "${PERCEPTION_CONFORMANCE}" "${PERCEPTION_FULL}" \
  "${PERCEPTION_CIP}" "${PERCEPTION_ELEMENTS}" \
  "${PERCEPTION_ROTA_NEGATIVE}" "${PERCEPTION_CIP_CONTRACT}" "${DIAGNOSTICS}"
CIP_FILE="${CIP_FILE:-/tmp/cip-validation-suite-compounds.smi}"
CIP_3D_FILE="${CIP_3D_FILE:-/tmp/cip-validation-suite-compounds-3d.sdf}"
CIP_SHA256="df178635c00b6c41fad820d2609fc4ff18403c63dec4e5c3e1756a6db5858059"
CIP_3D_SHA256="28a000b36506dabe5a45f6d7672451c61c6e34f3e0f027f2599000d850273325"
CIP_REVISION="6b9f9db46dadc6749da8234b05164e1e0fb413b9"
CIP_URL="https://raw.githubusercontent.com/CIPValidationSuite/ValidationSuite/${CIP_REVISION}/compounds.smi"
CIP_3D_URL="https://raw.githubusercontent.com/CIPValidationSuite/ValidationSuite/${CIP_REVISION}/compounds_3d.sdf"

if [[ ! -f "${CIP_FILE}" ]] ||
   [[ "$(sha256sum "${CIP_FILE}" | cut -d' ' -f1)" != "${CIP_SHA256}" ]]; then
    curl -fL "${CIP_URL}" -o "${CIP_FILE}"
fi
echo "${CIP_SHA256}  ${CIP_FILE}" | sha256sum -c -
if [[ ! -f "${CIP_3D_FILE}" ]] ||
   [[ "$(sha256sum "${CIP_3D_FILE}" | cut -d' ' -f1)" != "${CIP_3D_SHA256}" ]]; then
    curl -fL "${CIP_3D_URL}" -o "${CIP_3D_FILE}"
fi
echo "${CIP_3D_SHA256}  ${CIP_3D_FILE}" | sha256sum -c -

LOCAL_RUNNER="Experiment/Stereo/Canonicalization/configuration_free_local.py"
GLOBAL_ABC_RUNNER="Experiment/Stereo/Canonicalization/configuration_free_global.py"

python "${LOCAL_RUNNER}" internal --jobs "${BENCHMARK_JOBS}" --timeout 10 \
  --output "${CANON_LOCAL}/internal_local_canonicalization_report.json" \
  --table "${CANON_LOCAL}/internal_local_canonicalization_table.csv"

python "${LOCAL_RUNNER}" acs --jobs "${BENCHMARK_JOBS}" --timeout 10 \
  --output "${CANON_LOCAL}/acs_local_canonicalization_report.json" \
  --table "${CANON_LOCAL}/acs_local_canonicalization_table.csv"

python "${LOCAL_RUNNER}" cip \
  --cip-path "${CIP_FILE}" --jobs "${BENCHMARK_JOBS}" --timeout 10 \
  --output "${CANON_LOCAL}/cip_local_canonicalization_report.json" \
  --table "${CANON_LOCAL}/cip_local_canonicalization_table.csv"

python "${LOCAL_RUNNER}" rota --jobs "${BENCHMARK_JOBS}" --timeout 10 \
  --output "${CANON_LOCAL}/rota_local_canonicalization_report.json" \
  --table "${CANON_LOCAL}/rota_local_canonicalization_table.csv"

# Complete maintained configuration-free global matrix.
python "${GLOBAL_ABC_RUNNER}" A --cip-path "${CIP_FILE}" \
  --enumeration-mode formal --jobs "${BENCHMARK_JOBS}" \
  --timeout 10 --max-assignments 4096 --progress-interval 0 \
  --output "${CANON_ABC}/A/global_formal_canonicalization_report.json" \
  --table "${CANON_ABC}/A/global_formal_canonicalization_table.csv"

python "${GLOBAL_ABC_RUNNER}" A --cip-path "${CIP_FILE}" \
  --enumeration-mode raw --jobs "${BENCHMARK_JOBS}" \
  --timeout 10 --max-assignments 4096 --progress-interval 0 \
  --output "${CANON_ABC}/A/global_raw_canonicalization_report.json" \
  --table "${CANON_ABC}/A/global_raw_canonicalization_table.csv"

python "${GLOBAL_ABC_RUNNER}" B --cip-path "${CIP_FILE}" \
  --enumeration-mode formal --jobs "${BENCHMARK_JOBS}" \
  --timeout 10 --max-assignments 4096 --progress-interval 0 \
  --output "${CANON_ABC}/B/global_formal_canonicalization_report.json" \
  --table "${CANON_ABC}/B/global_formal_canonicalization_table.csv"

python "${GLOBAL_ABC_RUNNER}" C \
  --enumeration-mode formal --jobs "${BENCHMARK_JOBS}" \
  --timeout 10 --max-assignments 4096 --progress-interval 0 \
  --output "${CANON_ABC}/C/global_formal_canonicalization_report.json" \
  --table "${CANON_ABC}/C/global_formal_canonicalization_table.csv"

python Experiment/Stereo/Chirality/published.py \
  --output "${CHIRALITY_PUBLISHED}/published_acs_chirality_report.json"

python Experiment/Stereo/Chirality/run.py acs \
  --case-timeout 5 --identity-profile chemical \
  --output "${CHIRALITY_EXACT}/exact_acs_mirror_report.json"

python Experiment/Stereo/Chirality/run.py relations \
  --output "${CHIRALITY_RELATIONS}/stereoisomer_relation_report.json"

python Experiment/Stereo/Perception/axis_loci.py \
  --output "${PERCEPTION_AXIS}/rota_locus_report.json"

python Experiment/Stereo/Perception/conformance.py \
  --output "${PERCEPTION_CONFORMANCE}/perception_conformance_report.json"

python Experiment/Stereo/Perception/full_detection.py \
  --cip-path "${CIP_FILE}" \
  --case-timeout-seconds 30 \
  --output "${PERCEPTION_FULL}/full_detection_report.json"

python Experiment/Stereo/Perception/cip_labels.py \
  --cip-path "${CIP_FILE}" \
  --cip-3d-path "${CIP_3D_FILE}" \
  --output "${PERCEPTION_CIP}/cip_native_report.json"

python Experiment/Stereo/Perception/rota_synthetic_negatives.py \
  --output "${PERCEPTION_ROTA_NEGATIVE}/negative_axes.json"

python Experiment/Stereo/Perception/cip_input_contract.py \
  --cip-path "${CIP_FILE}" \
  --cip-3d-path "${CIP_3D_FILE}" \
  --output "${PERCEPTION_CIP_CONTRACT}/input_contract_audit.json"

python Experiment/Stereo/Perception/stereo_elements.py \
  --cip-path "${CIP_FILE}" \
  --output "${PERCEPTION_ELEMENTS}/stereo_element_report.json"

# This stage consumes the stereo-element report generated immediately above.
# Keeping it here makes a clean checkout independent of prior-run outputs.
python Experiment/Stereo/Canonicalization/atom_relabel.py global-local \
  --permutations 8 --class-relabelings 3 --exhaustive-max-atoms 5 \
  --timeout 10 \
  --output "${CANON_GLOBAL}/global_local_canonicalization_report.json" \
  --inventory-json "${CANON_GLOBAL}/canonicalization_inventory.json" \
  --inventory-csv "${CANON_GLOBAL}/canonicalization_inventory.csv"

python - \
  "${CANON_LOCAL}/canonicalization_inventory.json" \
  "${CANON_LOCAL}/canonicalization_inventory.csv" <<'PY'
import sys
from pathlib import Path

from Experiment.Stereo.Canonicalization.inventory import (
    build_inventory,
    write_inventory,
)

write_inventory(
    build_inventory(),
    json_path=Path(sys.argv[1]),
    csv_path=Path(sys.argv[2]),
)
PY

python Experiment/Stereo/datasets.py \
  --max-isomers 256 --case-timeout-seconds 10 \
  --cip-path "${CIP_FILE}" \
  --output "${DIAGNOSTICS}/benchmark_report.json"

python - "${CANON_MULTI}/multi_element_canonicalization_report.json" <<'PY'
import json
import sys
from pathlib import Path

from Experiment.Stereo.Canonicalization.multi_element import (
    benchmark_multi_element_canonicalization,
)

output = Path(sys.argv[1])
report = benchmark_multi_element_canonicalization(
    synthetic_relabelings=1,
    acs_relabelings=2,
    timeout_seconds=10.0,
)
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"output": str(output), "passed": report["passed"]}, indent=2))
PY

python - \
  "${REPOSITORY_ROOT}/Experiment/Stereo/Data/manifest.json" \
  "${PERCEPTION_FULL}/full_detection_report.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
report_path = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["empirical_benchmarks"][
    "full_configuration_neutral_carrier_detection"
]["sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
manifest_path.write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)
PY

# Requested raw stress products. These are intentionally last because they may
# require several days even with case-level parallelism.
python "${GLOBAL_ABC_RUNNER}" B --cip-path "${CIP_FILE}" \
  --enumeration-mode raw --allow-expensive-raw \
  --jobs "${BENCHMARK_JOBS}" \
  --timeout 10 --max-assignments 4096 --progress-interval 60 \
  --output "${CANON_ABC}/B/global_raw_canonicalization_report.json" \
  --table "${CANON_ABC}/B/global_raw_canonicalization_table.csv"

python "${GLOBAL_ABC_RUNNER}" C --case-id VS146 \
  --enumeration-mode raw --allow-expensive-raw \
  --jobs "${BENCHMARK_JOBS}" \
  --timeout 10 --max-assignments 4096 --progress-interval 60 \
  --output "${CANON_ABC}/C/global_raw_vs146_canonicalization_report.json" \
  --table "${CANON_ABC}/C/global_raw_vs146_canonicalization_table.csv"

echo "Stereo benchmark reports, including raw B and raw C/VS146, were written."
