#!/usr/bin/env python
"""Verify exported selected-case certificates without accessing SynEPD."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.MTG.verified_cases import _unpack_certificate  # noqa: E402


def verify(path):
    report = json.loads(path.read_text())
    if report["schema"] != "synkit.mtg-verified-cases/1":
        raise ValueError("Unsupported exported-case schema.")
    certificates = 0
    for case in report["cases"]:
        for interpretation in case["interpretations"]:
            for value in interpretation["certificates"]:
                certificate = _unpack_certificate(value)
                replay = certificate.replay()
                if not replay.valid:
                    raise ValueError(
                        f"Invalid record {case['id']} certificate: {replay.issues}"
                    )
                certificates += 1
            if (
                interpretation["mtg_queries"]
                != interpretation["full_trajectory_queries"]
            ):
                raise ValueError("Stored baseline query results disagree.")
    return {
        "certificates_verified": certificates,
        "cases": len(report["cases"]),
        "scope": "Certificate equations independently checked; stored query equality checked, "
        "not an independent chemical oracle.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.evidence), indent=2))
