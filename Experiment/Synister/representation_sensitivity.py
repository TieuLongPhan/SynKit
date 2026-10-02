"""R2 alternative bond encoding and exact numeric shells (original labels)."""
from dataclasses import asdict
from fractions import Fraction

from rdkit import Chem

from synkit.Chem.Mapper.identifiability import Endpoint, extract_label, parse_endpoint
from synkit.Chem.Mapper.numerical_scope import validate_study_domain


def kekule_endpoint(smiles, order=None):
    original = parse_endpoint(smiles)
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles, params))
    if order is not None:
        order = list(order)
        if sorted(order) != list(range(mol.GetNumAtoms())):
            raise ValueError("Atom ordering must be a permutation")
        mol = Chem.RenumberAtoms(mol, order)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    if order is not None:
        mol = Chem.RenumberAtoms(mol, [order.index(i) for i in range(len(order))])
    alternative = Endpoint(
        tuple(a.GetAtomicNum() for a in mol.GetAtoms()),
        tuple(a.GetFormalCharge() for a in mol.GetAtoms()),
        tuple(a.GetTotalNumHs() for a in mol.GetAtoms()),
        tuple(sorted((*sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())),
                      int(2*b.GetBondTypeAsDouble())) for b in mol.GetBonds())))
    if (original.atomic_numbers, original.charges, original.hcounts) != (
            alternative.atomic_numbers, alternative.charges, alternative.hcounts):
        raise ValueError("Kekulization changed atom attributes or indices")
    if {(i,j) for i,j,_ in original.bonds} != {(i,j) for i,j,_ in alternative.bonds}:
        raise ValueError("Kekulization changed bond presence")
    return original, alternative


def ordering_control(smiles):
    """Reverse atom order before encoding, then transport back to original indices.

    This records encoding dependence, not invariance of every resonance form.
    It never replaces the encoding used by the scientific search.
    """
    original, forward = kekule_endpoint(smiles)
    order = list(reversed(range(len(original.atomic_numbers))))
    _, reverse = kekule_endpoint(smiles, order)
    return {"permutation":order, "encoding_changed":forward != reverse,
            "forward":asdict(forward), "transported_reverse":asdict(reverse)}


def weighted_cost(r, p, mapping):
    n = len(r.atomic_numbers)
    if sorted(mapping) != list(range(n)) or any(
            r.atomic_numbers[i] != p.atomic_numbers[mapping[i]] for i in range(n)):
        raise ValueError("Not an element-compatible bijection")
    a = {(i,j): w for i,j,w in r.bonds}
    b = {(i,j): w for i,j,w in p.bonds}
    return Fraction(sum(abs(a.get((i,j),0) - b.get(tuple(sorted((mapping[i],mapping[j]))),0))
                        for i in range(n) for j in range(i+1,n)), 2)


def search(r, p, *, objective_endpoints=None, target="minimal",
           initial_mapping=None, seconds=60, max_mappings=100000):
    from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings
    a,b = objective_endpoints if objective_endpoints is not None else (r,p)
    for original, alternative in ((r,a),(p,b)):
        if (original.atomic_numbers, original.charges, original.hcounts) != (
                alternative.atomic_numbers, alternative.charges, alternative.hcounts):
            raise ValueError("Alternative endpoint attributes differ")
        if {(i,j) for i,j,_ in original.bonds} != {(i,j) for i,j,_ in alternative.bonds}:
            raise ValueError("Alternative endpoint bond presence differs")
    domain = validate_study_domain(a,b)
    result = enumerate_distance_mappings(
        [a.graph(), b.graph()], CD=target, binary=False, symmetry_pruning=False,
        max_bijections=None, max_mappings=max_mappings, initial_mapping=initial_mapping,
        time_limit_seconds=seconds, compute_minimum_cost=target == "minimal")
    complete = result.complete and result.status in ("complete", "no_solutions")
    labels, joint = {}, {}
    expected = result.cost if target == "minimal" else target
    if complete:
        for mapping in result.mappings:
            if weighted_cost(a,b,mapping) != Fraction(expected):
                raise ValueError("Independent weighted objective disagrees with search")
            value = extract_label(r,p,mapping)
            record = {"mapping":list(mapping), "label":asdict(value)}
            labels.setdefault(tuple(sorted(value.changed_bonds)), record)
            joint.setdefault(value, record)
    return {"status":"complete" if complete else "unresolved",
            "target":target, "objective":"weighted_alternative" if objective_endpoints else "weighted_original",
            "label_endpoints":"original_weighted", "numerical_domain":domain,
            "minimum":result.cost if target == "minimal" else None,
            "enumeration_complete":complete, "labels_complete":complete,
            "joint_labels_complete":complete, "empty":complete and not result.mappings,
            "solver_status":result.status, "truncation_reason":result.truncation_reason,
            "emitted_representatives":len(result.mappings), "symmetry_pruning":False,
            "labels":[labels[k] for k in sorted(labels)], "joint_labels":list(joint.values())}


def union_labels(minimum, shells):
    """An incomplete shell must never masquerade as an empty shell."""
    if len(shells) != 2:
        raise ValueError("R2 requires exactly two fixed offset shells")
    if any(x["status"] != "complete" or not x.get("labels_complete",False)
           for x in (minimum,*shells)):
        return {"status":"unresolved", "labels_complete":False, "labels":[]}
    labels = {}
    for result in (minimum,*shells):
        for record in result["labels"]:
            key = tuple(tuple(x) for x in record["label"]["typed_bond_edits"])
            labels.setdefault(key,record)
    return {"status":"complete", "labels_complete":True, "labels":list(labels.values())}
