"""
models.py

Pydantic models and helpers for motif config validation and normalization.

This file encodes the config vocabulary and *documents* the CRN / systems-biology
semantics of each config type (limited, rate_limited, immediate_sink, export_after_T, ...).

Summary: the config vocabulary maps directly onto widely used constructs in CRN / systems
biology research:

 - "limited"  -> closed batch / finite initial condition.  Use when you want a conserved
                resource that can be exhausted.
                (Gillespie: set initial count N0; ODE: initial condition x(0)=N0;
                 CRNT: contributes to conservation laws.)

 - "rate_limited" -> chemostat / continuous inflow / reservoir. Modeled as Ø -> X
                     with propensity k (stochastic) or source term +k (ODE). Widely used
                     for nutrient uptake in metabolic models and synthetic biology.

 - "immediate_sink" -> open boundary / perfect absorber. Modeled as X -> Ø with a large
                       rate or simply treated as removed upon production. Useful for
                       exported waste or product removal.

 - "export_after_T" -> buffered export / delayed removal / secretion. Implement either
                       (a) explicit buffer pool with rules for transfer when threshold/time
                       reached, or (b) delayed reaction (X -> Ø after T). Delay stochastic
                       simulation methods (DSSA) or hybrid deterministics are common.

 - "buffered_sink" -> finite-capacity collector (tracking occupancy). Useful in queueing
                      analogies and when capacity-limited secretion occurs.

 - "recycle" -> partial return of mass to system. Model as X -> Ø with rate k plus Ø -> Y
                 with fraction f, or explicit X -> (1-f)*Waste + f*Recycled.

 - "repairable"/"inactive" -> inactive states and repair dynamics (two-state models),
                              common in protein folding/repair and synthetic circuits.

 - "pulsed"/"burst"/"stochastic" -> time-dependent or distributed arrivals; used in experiments
                                    with pulses (induction) or transcriptional bursting models.

Engine translation hints (illustrative) -- adapt to simulation backend:
 - For each normalized source/sink, here's how to map to primitives:
   Sources:
    - limited: set initial condition x(0) = initial
        * ODE: x(0)=initial
        * SSA: populate state vector with initial count

    - rate_limited ({"rate": k}):
        * SSA: add reaction R: Ø -> X, propensity a = k  (or a = k * volume scaling)
        * ODE: dx/dt += k

    - pulsed (period T, pulse_size s):
        * implement time-dependent propensity or explicit events at t = n*T adding s counts

    - burst (one-time):
        * SSA: at t0 add burst_size to state
        * ODE: add step at t0

   Sinks:
    - immediate_sink:
        * SSA: X -> Ø with rate k (or infinite -> treat removal instantaneous when produced)
        * ODE: dx/dt -= k*x  (for first-order) or remove immediately in discrete transitions

    - export_after_T (delay_T, threshold):
        * implement buffer B; X -> B (fast), then when B.size >= threshold or t >= delay_T
          move B -> Exported. For stochastic delays, use delay-SSA or queue with timestamps.

    - recycle: X -> Ø with fraction (1-f) to waste and f to recycled species (explicit reactions)

    - repairable/inactive: model explicitly as X <-> X_inactive with rates k_off/k_on

Research compatibility notes (short):
 1) CRN Theory (Feinberg etc.): Definitions create open CRNs (sources/sinks). Finite
    "limited" resources lead to conserved linear combinations; check for stoichiometric
    conservation if want to use deficiency / steady-state theory.

 2) Stochastic vs Deterministic: many papers compare SSA vs ODE for motifs like 1-3. Use
    SSA for low copy numbers or bursting; use ODE for large-copy deterministic approximations.

 3) Enzyme kinetics: model explicit enzyme:substrate complexes or use Michaelis-Menten reductions
    (valid in quasi-steady-state regimes) for `Cat.*` or `Enz_*` motifs.

 4) Delays & buffers: secretion/export workflows (export_after_T) are standard in gene expression
    literature — use delay-SSA or buffer pools for faithful dynamics.

 5) Chemostat / metabolic inflow: `rate_limited` corresponds to chemostat-like nutrient supply;
    parameterize rates to match experimental dilution/feeding rates (units consistent with engine).

Implementation notes:
 - Accept both shorthand ("limited") and full dicts {"type": "limited", "initial": 10}.
 - Provide clear validation errors (pydantic is used here).
 - Keep units explicit in comments and YAML (events/sec vs concentration).
"""

from pydantic import BaseModel, Field, ValidationError
from typing import Literal, Dict, Any, Optional, Type


# --- Source models ----------------------------------------------------------
class SourceBase(BaseModel):
    type: str


class LimitedSource(SourceBase):
    type: Literal["limited"] = "limited"
    initial: int = Field(1, ge=0, description="Initial copy number or concentration")


class RateLimitedSource(SourceBase):
    type: Literal["rate_limited"] = "rate_limited"
    rate: float = Field(..., gt=0, description="Arrival rate (events / time)")
    distribution: Optional[str] = Field("poisson", description="arrival distribution")


class UnlimitedSource(SourceBase):
    type: Literal["unlimited"] = "unlimited"


class PulsedSource(SourceBase):
    type: Literal["pulsed"] = "pulsed"
    period: float = Field(..., gt=0)
    pulse_size: int = Field(1, ge=1)
    start_time: float = Field(0.0, ge=0.0)


class BurstSource(SourceBase):
    type: Literal["burst"] = "burst"
    burst_size: int = Field(..., ge=1)
    t0: float = Field(0.0, ge=0.0)


class SeedSource(SourceBase):
    type: Literal["seed"] = "seed"
    initial: int = Field(1, ge=0)


class StochasticSource(SourceBase):
    type: Literal["stochastic"] = "stochastic"
    lambda_: float = Field(1.0, alias="lambda")


# mapping from type string -> model class
SOURCE_MODEL_MAP: Dict[str, Type[SourceBase]] = {
    "limited": LimitedSource,
    "rate_limited": RateLimitedSource,
    "unlimited": UnlimitedSource,
    "pulsed": PulsedSource,
    "burst": BurstSource,
    "seed": SeedSource,
    "stochastic": StochasticSource,
}


# --- Sink models ------------------------------------------------------------
class SinkBase(BaseModel):
    type: str


class ImmediateSink(SinkBase):
    type: Literal["immediate_sink"] = "immediate_sink"


class BufferedSink(SinkBase):
    type: Literal["buffered_sink"] = "buffered_sink"
    capacity: int = Field(100, gt=0)


class ExportAfterT(SinkBase):
    type: Literal["export_after_T"] = "export_after_T"
    delay_T: float = Field(0.0, ge=0.0)
    threshold: int = Field(1, ge=1)


class RecycleSink(SinkBase):
    type: Literal["recycle"] = "recycle"
    recycle_fraction: float = Field(0.5, ge=0.0, le=1.0)


class RepairableSink(SinkBase):
    type: Literal["repairable"] = "repairable"
    repair_rate: float = Field(0.1, ge=0.0)


class WastePool(SinkBase):
    type: Literal["waste_pool"] = "waste_pool"


class InactiveSink(SinkBase):
    type: Literal["inactive"] = "inactive"
    repair_rate: float = Field(0.01, ge=0.0)


class PermeableSink(SinkBase):
    type: Literal["permeable"] = "permeable"
    permeability_rate: float = Field(0.01, ge=0.0)


SINK_MODEL_MAP: Dict[str, Type[SinkBase]] = {
    "immediate_sink": ImmediateSink,
    "buffered_sink": BufferedSink,
    "export_after_T": ExportAfterT,
    "recycle": RecycleSink,
    "repairable": RepairableSink,
    "waste_pool": WastePool,
    "inactive": InactiveSink,
    "permeable": PermeableSink,
}


# --- MotifConfig -----------------------------------------------------------
class MotifConfig(BaseModel):
    """
    Canonical representation of a motif runtime config.

    - `regime`: 'stochastic' | 'deterministic' | 'hybrid'
    - `sources`: mapping name -> SourceBase-derived instance
    - `sinks`: mapping name -> SinkBase-derived instance
    - `meta`: optional dict for arbitrary metadata
    """

    regime: Literal["stochastic", "deterministic", "hybrid"] = "stochastic"
    sources: Dict[str, SourceBase] = {}
    sinks: Dict[str, SinkBase] = {}
    meta: Optional[Dict[str, Any]] = None

    @classmethod
    def from_raw(cls, raw: Optional[Dict[str, Any]]):
        """
        Construct MotifConfig from raw dict loaded from YAML/JSON.

        Accepts shorthand: the values in `sources` and `sinks` may be strings (type)
        or dicts containing a `type` key and parameters.
        """
        raw = raw or {}
        regime = raw.get("regime", "stochastic")
        raw_sources = raw.get("sources", {}) or {}
        raw_sinks = raw.get("sinks", {}) or {}
        meta = raw.get("meta")

        parsed_sources: Dict[str, SourceBase] = {}
        parsed_sinks: Dict[str, SinkBase] = {}

        # parse sources
        for name, spec in raw_sources.items():
            parsed_sources[name] = parse_source_spec(spec)

        for name, spec in raw_sinks.items():
            parsed_sinks[name] = parse_sink_spec(spec)

        return cls(regime=regime, sources=parsed_sources, sinks=parsed_sinks, meta=meta)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "sources": {k: v.dict(by_alias=True) for k, v in self.sources.items()},
            "sinks": {k: v.dict() for k, v in self.sinks.items()},
            "meta": self.meta,
        }


# --- parsing helpers -------------------------------------------------------
def _ensure_type_dict(spec: Any) -> Dict[str, Any]:
    """Turn shorthand specs into dicts with a 'type' key."""
    if isinstance(spec, str):
        return {"type": spec}
    if isinstance(spec, dict):
        if "type" not in spec:
            raise ValueError(f"Spec dict missing 'type' key: {spec}")
        return dict(spec)
    raise TypeError(f"Unsupported spec type: {type(spec)}")


def parse_source_spec(spec: Any) -> SourceBase:
    s = _ensure_type_dict(spec)
    t = s["type"]
    model_cls = SOURCE_MODEL_MAP.get(t)
    if model_cls is None:
        # fallback: create a minimal SourceBase
        return SourceBase(type=t)
    # adapt alias 'lambda' to 'lambda_' if needed
    if t == "stochastic" and "lambda" in s:
        s["lambda"] = s.pop("lambda")
    try:
        return model_cls.parse_obj(s)
    except ValidationError as e:
        raise ValidationError(f"Failed to parse source spec for type {t}: {e}")


def parse_sink_spec(spec: Any) -> SinkBase:
    s = _ensure_type_dict(spec)
    t = s["type"]
    model_cls = SINK_MODEL_MAP.get(t)
    if model_cls is None:
        return SinkBase(type=t)
    try:
        return model_cls.parse_obj(s)
    except ValidationError as e:
        raise ValidationError(f"Failed to parse sink spec for type {t}: {e}")
