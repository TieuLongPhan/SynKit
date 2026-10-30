"""Stable directed glycolysis inputs for prescribed-route controls."""

import json
from pathlib import Path

from synkit.CRN import SynCRN

DEFAULT_FIXTURE = Path(__file__).resolve().parents[1] / "data/glycolysis_routes.json"


def load_glycolysis(path=None):
    """Load equations and preserve their labels for prescribed reaction flows."""
    path = DEFAULT_FIXTURE if path is None else Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "synkit.crn-glycolysis-input/1":
        raise ValueError(f"{path}: unsupported glycolysis input schema")
    equations = payload["equations"]
    if not equations or not all(
        isinstance(e, str) and ">>" in e for e in equations.values()
    ):
        raise ValueError(f"{path}: expected nonempty directed reaction equations")
    model = SynCRN.from_reaction_strings(list(equations.values()))
    if len(model.reactions) != len(equations):
        raise ValueError(f"{path}: reaction labels do not map one-to-one to reactions")
    for reaction, label in zip(model.reactions.values(), equations):
        reaction.label = label
    return model, equations
