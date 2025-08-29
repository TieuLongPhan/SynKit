"""
loader.py

Loader utilities to read YAML config files, validate via models.py, normalize,
and translate to engine-friendly primitives (e.g., SSA reaction tuples).

We include detailed documentation of how each config type should be interpreted
by simulators (Gillespie SSA, ODE, delay-aware solvers), plus practical tips
for unit consistency, regime selection, and CRN-theory checks.

SSA reaction tuple format used here (simple convenience representation):
    (reactants: Dict[str,int], products: Dict[str,int], rate: float, meta: Dict[str,Any])

"""

from pathlib import Path
from typing import Dict, Any, Tuple, List
import yaml

from models import MotifConfig


def load_yaml(path: Path) -> dict:
    """
    Load YAML file and return dict. Returns {} on empty file.
    """
    text = Path(path).read_text()
    return yaml.safe_load(text) or {}


def load_and_validate_configs(path: Path) -> Dict[str, MotifConfig]:
    """
    Load YAML configs and convert each motif's dict into MotifConfig instances.

    Returns a dict mapping motif_name -> MotifConfig.
    Raises RuntimeError if any motif config is invalid.
    """
    raw = load_yaml(Path(path))
    parsed: Dict[str, MotifConfig] = {}
    for motif_name, motif_cfg in raw.items():
        try:
            mc = MotifConfig.from_raw(motif_cfg)
            parsed[motif_name] = mc
        except Exception as e:
            raise RuntimeError(f"Invalid config for motif {motif_name}: {e}")
    return parsed


def normalized_all_motifs(
    all_motifs: Dict[str, Tuple[List[str], Dict[str, Any]]],
    configs: Dict[str, MotifConfig],
) -> Dict[str, Tuple[List[str], MotifConfig]]:
    """
    Return mapping motif_name -> (reactions:list, MotifConfig).

    If a motif has no entry in `configs`, an empty MotifConfig (defaults) is used.
    """
    out: Dict[str, Tuple[List[str], MotifConfig]] = {}
    for name, (reactions, raw_cfg) in all_motifs.items():
        cfg = configs.get(name)
        if cfg is None:
            cfg = MotifConfig.from_raw({})
        out[name] = (list(reactions), cfg)
    return out


# Helper: strip 'Source.' prefix when producing species that come from a Source.* entry
def _strip_source_prefix(name: str) -> str:
    return name.replace("Source.", "") if name.startswith("Source.") else name


def config_to_ssa(
    motif_name: str, reactions: List[str], motif_config: MotifConfig
) -> List[Tuple[Dict[str, int], Dict[str, int], float, Dict[str, Any]]]:
    """
    Translate MotifConfig into a list of SSA-like reactions.

    Notes:
      - limited/seed: returned as init markers (rate=0 and meta['init']=N)
      - rate_limited: Ø -> X with given rate
      - immediate_sink: X -> Ø with very large rate (or engine-specific sentinel)
      - export_after_T, buffered_sink: produce buffer transition entries with metadata

    Output tuple: (reactants, products, rate, meta)
    - reactants/products: dict species->stoichiometry
    - rate: float (0.0 for placeholders/markers; engines can interpret meta)
    - meta: hints for engine (init, pulsed, buffer, threshold, delay_T)
    """
    ssa: List[Tuple[Dict[str, int], Dict[str, int], float, Dict[str, Any]]] = []

    # sources: create Ø -> X reactions or init markers
    for src_name, src_model in motif_config.sources.items():
        species = _strip_source_prefix(src_name)
        t = src_model.type
        if t == "rate_limited":
            rate = float(getattr(src_model, "rate", 1.0))
            # Add explicit zero->X reaction with rate
            ssa.append(
                ({}, {species: 1}, rate, {"origin": "rate_limited", "source": src_name})
            )
        elif t in ("limited", "seed"):
            initial = int(getattr(src_model, "initial", 1))
            # init marker: engine should set state[species] = initial
            ssa.append(
                (
                    {},
                    {species: 0},
                    0.0,
                    {"init": initial, "origin": t, "source": src_name},
                )
            )
        elif t == "pulsed":
            ssa.append(
                (
                    {},
                    {species: int(getattr(src_model, "pulse_size", 1))},
                    0.0,
                    {
                        "pulsed": True,
                        "period": getattr(src_model, "period"),
                        "start_time": getattr(src_model, "start_time", 0.0),
                    },
                )
            )
        elif t == "burst":
            ssa.append(
                (
                    {},
                    {species: int(getattr(src_model, "burst_size", 1))},
                    0.0,
                    {"burst_at": getattr(src_model, "t0")},
                )
            )
        elif t == "unlimited":
            # treat as constant source with effectively infinite supply; engine model as rate-limited with large rate
            ssa.append(({}, {species: 1}, 0.0, {"origin": "unlimited"}))
        else:
            # fallback: record type for engine to interpret
            ssa.append(({}, {species: 1}, 0.0, {"origin": t}))

    # sinks: create X -> Ø or buffer transitions
    for sink_name, sink_model in motif_config.sinks.items():
        t = sink_model.type
        # If the sink_name looks like a species within the motif, use it as-is.
        if t == "immediate_sink":
            # use a large effective rate placeholder; engine should map this appropriately
            ssa.append(({sink_name: 1}, {}, 1e6, {"sink_type": "immediate"}))
        elif t == "export_after_T":
            buff = f"{sink_name}_buffer"
            ssa.append(
                (
                    {sink_name: 1},
                    {buff: 1},
                    0.0,
                    {
                        "buffer": True,
                        "threshold": getattr(sink_model, "threshold", 1),
                        "delay_T": getattr(sink_model, "delay_T", 0.0),
                    },
                )
            )
        elif t == "buffered_sink":
            pool = f"{sink_name}_pool"
            ssa.append(
                (
                    {sink_name: 1},
                    {pool: 1},
                    0.0,
                    {
                        "buffered": True,
                        "capacity": getattr(sink_model, "capacity", 100),
                    },
                )
            )
        elif t == "recycle":
            frac = getattr(sink_model, "recycle_fraction", 0.5)
            # Engines should implement split semantics. Here we add a meta entry to instruct engine.
            ssa.append(
                (
                    {sink_name: 1},
                    {},
                    0.0,
                    {"sink_type": "recycle", "recycle_fraction": frac},
                )
            )
        elif t == "repairable":
            # Suggest modeling explicit X <-> X_inactive with rates; provide hints
            ssa.append(
                (
                    {sink_name: 1},
                    {},
                    0.0,
                    {
                        "sink_type": "repairable",
                        "repair_rate": getattr(sink_model, "repair_rate", 0.1),
                    },
                )
            )
        else:
            # generic sink marker
            ssa.append(({sink_name: 1}, {}, 0.0, {"sink_type": t}))

    return ssa
