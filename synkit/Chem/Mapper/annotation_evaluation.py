"""Secondary label metrics, structural classes and shared annotation policies.

Inputs must retain complete joint labels, not merely one representative per
changed-bond set. Reference annotations are revealed only to this evaluation
stage; they are never used to generate predictions or candidate sets.
"""

import json
from numbers import Integral

from .evaluation import IncompleteSymmetry, ScoreWitness, bond_f1
from .identifiability import extract_label
from .orbit_evaluation import SupportOrbitEvaluator
from .slap.lap import _adjacency_and_elements
from .spectrum import exact_its_and_template_codes


def features(label, kind):
    if kind == "bond":
        return frozenset(("bond", i, j) for i, j in label.changed_bonds)
    if kind == "atom":
        return frozenset(("atom", i) for i in label.centre_atoms)
    if kind == "typed":
        return frozenset(("bond", *edit) for edit in label.typed_bond_edits)
    if kind == "joint":
        return features(label, "typed") | frozenset(
            ("unary", *edit) for edit in label.unary_changes
        )
    raise ValueError("Unknown label kind")


class FeatureOrbitEvaluator(SupportOrbitEvaluator):
    """The same exact support action on typed bonds, unary edits or atom sets."""

    def __init__(self, reactant, *, exact_match=False, **limits):
        super().__init__(reactant, **limits)
        self.exact_match = exact_match

    def _validate(self, label):
        label = frozenset(tuple(item) for item in label)
        for item in label:
            if not item or item[0] not in {"atom", "bond", "unary"}:
                raise ValueError("Unknown feature token")
            if item[0] == "atom" and len(item) == 2:
                indices = item[1:2]
            elif item[0] == "bond" and len(item) in (3, 5):
                indices = item[1:3]
                if not item[1] < item[2]:
                    raise ValueError("Bond coordinates must be ordered")
                if len(item) == 5 and (
                    any(not isinstance(x, Integral) for x in item[3:])
                    or item[3] == item[4]
                ):
                    raise ValueError(
                        "Typed edits need distinct integer before/after orders"
                    )
            elif item[0] == "unary" and len(item) == 5:
                indices = item[1:2]
                if item[2] not in {"charges", "hcounts"} or any(
                    not isinstance(x, Integral) for x in item[3:]
                ):
                    raise ValueError("Unsupported unary edit")
            else:
                raise ValueError("Invalid feature shape")
            if any(not isinstance(i, Integral) or not 0 <= i < self.n for i in indices):
                raise ValueError("Out-of-range feature coordinate")
        return label

    def _support(self, label):
        return tuple(
            sorted({i for x in label for i in (x[1:3] if x[0] == "bond" else x[1:2])})
        )

    def _transport(self, label, permutation):
        result = []
        for item in label:
            if item[0] == "bond":
                i, j = sorted((permutation[item[1]], permutation[item[2]]))
                result.append(("bond", i, j, *item[3:]))
            else:
                result.append((item[0], permutation[item[1]], *item[2:]))
        return frozenset(result)

    def score(self, prediction, label):
        prediction = self._validate(prediction)
        best = None
        for image, witness in self.orbit(label).items():
            self._check()
            value = (
                int(prediction == image)
                if self.exact_match
                else bond_f1(prediction, image)
            )
            if best is None or value > best.score:
                best = ScoreWitness(value, witness)
        return best


def _witness(value):
    return {
        "difference": str(value.difference),
        "label": sorted(value.label),
        "a_score": str(value.a.score),
        "b_score": str(value.b.score),
        "a_transporter": value.a.transporter,
        "b_transporter": value.b.transporter,
    }


def analyze_annotations(  # noqa: C901
    reactant,
    product,
    candidate_mappings,
    prediction_a,
    prediction_b,
    *,
    joint_labels_complete,
    minimum,
    reference_mapping=None,
    metric_seconds=10,
    canonical_seconds=0.25,
    template_radius=1,
):
    """Evaluate complete joint witnesses with independently reported stages.

    Canonical full ITS codes choose a deterministic class convention. Policies
    nearest to A/B use primary bond F1 with that canonical code as tie-break;
    every method is evaluated against the resulting *shared* label. These
    conventions are diagnostics, not chemical ground truth.
    """
    if not joint_labels_complete:
        raise ValueError("Secondary evaluation requires complete joint labels")
    mappings = [tuple(m) for m in candidate_mappings]
    if not mappings:
        raise ValueError("A nonempty candidate set is required")
    labels = [extract_label(reactant, product, m) for m in mappings]
    if any(x.weighted_distance != minimum for x in labels):
        raise ValueError(
            "Candidate objective does not match the declared weighted minimum"
        )
    pa = extract_label(reactant, product, prediction_a)
    pb = extract_label(reactant, product, prediction_b)
    reference = (
        None
        if reference_mapping is None
        else extract_label(reactant, product, reference_mapping)
    )
    output = {"status": "evaluated", "metrics": {}, "reference": {}, "policies": {}}
    if reference is not None:
        output["reference"] = {
            "weighted_distance": str(reference.weighted_distance),
            "mapping_in_minimum": reference.weighted_distance == minimum,
        }
    primary_scores, primary_keys = None, None
    for name, kind, exact in (
        ("bond_f1", "bond", False),
        ("atom_f1", "atom", False),
        ("typed_f1", "typed", False),
        ("bond_exact", "bond", True),
        ("joint_exact", "joint", True),
    ):
        try:
            scorer = FeatureOrbitEvaluator(
                reactant, exact_match=exact, time_limit_seconds=metric_seconds
            )
            candidates = [features(x, kind) for x in labels]
            a, b = features(pa, kind), features(pb, kind)
            lo, hi = scorer.paired_envelope(a, b, candidates, labels_complete=True)
            keys = [scorer.orbit_key(x) for x in candidates]
            values = [
                (scorer.score(a, x).score, scorer.score(b, x).score) for x in candidates
            ]
            record = {
                "status": "complete",
                "lower": _witness(lo),
                "upper": _witness(hi),
                "width": str(hi.difference - lo.difference),
                "fixed_labels": len(set(candidates)),
                "label_orbits": len(set(keys)),
            }
            output["metrics"][name] = record
            if name == "bond_f1":
                primary_scores, primary_keys = values, keys
            if reference is not None:
                try:
                    ref = features(reference, kind)
                    membership = scorer.orbit_key(ref) in set(keys)
                    ref_scores = {
                        "a": str(scorer.score(a, ref).score),
                        "b": str(scorer.score(b, ref).score),
                    }
                    record.update(
                        reference_status="complete",
                        reference_orbit_in_minimum=membership,
                        reference_scores=ref_scores,
                    )
                except IncompleteSymmetry as exc:
                    record.update(
                        reference_status="unresolved", reference_reason=str(exc)
                    )
        except IncompleteSymmetry as exc:
            output["metrics"][name] = {"status": "unresolved", "reason": str(exc)}
    rmatrix, elements = _adjacency_and_elements(reactant.graph(), False)
    pmatrix, _ = _adjacency_and_elements(product.graph(), False)
    properties = {
        name: (getattr(reactant, name), getattr(product, name))
        for name in ("charges", "hcounts")
    }
    codes = []
    for mapping in mappings:
        full, template, reason = exact_its_and_template_codes(
            rmatrix,
            pmatrix,
            elements,
            properties,
            mapping,
            template_radius=template_radius,
            timeout_seconds=canonical_seconds,
        )
        if reason is not None:
            output["structure"] = {"status": "unresolved", "reason": reason}
            break
        codes.append(
            (
                json.dumps(full, separators=(",", ":")),
                json.dumps(template, separators=(",", ":")),
            )
        )
    else:
        output["structure"] = {
            "status": "complete",
            "its_classes": len({x[0] for x in codes}),
            "template_classes": len({x[1] for x in codes}),
            "template_radius": template_radius,
            "codes": [
                {"mapping": m, "its_code": code[0], "template_code": code[1]}
                for m, code in zip(mappings, codes)
            ],
        }
        joint = output["metrics"]["joint_exact"]
        if (
            joint["status"] == "complete"
            and joint["label_orbits"] != output["structure"]["its_classes"]
        ):
            raise ValueError("Full ITS and joint-label orbit counts disagree")
        if primary_scores is not None:
            code_labels = {}
            for code, key in zip(codes, primary_keys):
                if code[0] in code_labels and code_labels[code[0]] != key:
                    raise ValueError(
                        "One ITS class induced multiple primary bond orbits"
                    )
                code_labels[code[0]] = key
            choices = {
                "canonical_its": min(range(len(codes)), key=lambda i: codes[i][0]),
                "nearest_a": min(
                    range(len(codes)),
                    key=lambda i: (-primary_scores[i][0], codes[i][0]),
                ),
                "nearest_b": min(
                    range(len(codes)),
                    key=lambda i: (-primary_scores[i][1], codes[i][0]),
                ),
            }
            for name, index in choices.items():
                a, b = primary_scores[index]
                output["policies"][name] = {
                    "mapping": mappings[index],
                    "its_code": codes[index][0],
                    "a_score": str(a),
                    "b_score": str(b),
                    "difference": str(a - b),
                }
    return output
