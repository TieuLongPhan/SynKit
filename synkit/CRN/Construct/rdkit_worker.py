"""RDKit C++ reaction application for unmapped SMARTS-generated CRN inputs."""

from functools import lru_cache
from itertools import permutations
from rdkit import Chem
from rdkit.Chem import rdChemReactions


@lru_cache(maxsize=128)
def _reaction(template):
    reaction = rdChemReactions.ReactionFromSmarts(template)
    if reaction is None:
        raise ValueError("Invalid RDKit reaction SMARTS")
    reaction.Initialize()
    return reaction


def _canonical_product(outcome, explicit_h):
    parts = []
    for molecule in outcome:
        Chem.SanitizeMol(molecule)
        if not explicit_h:
            molecule = Chem.RemoveHs(molecule)
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        parts.append(Chem.MolToSmiles(molecule, canonical=True))
    return ".".join(sorted(parts))


def apply_rdkit_rule_worker(args):
    """Apply templates in RDKit's native matcher; return canonical unmapped products.

    Atom-map-preserving output and SynReactor-specific strategies are outside
    this adapter's supported semantics. Product truncation raises explicitly.
    """
    index, rule, substrate, explicit_h, implicit_temp, strategy, keys = args
    if implicit_temp or strategy is not None:
        raise ValueError(
            "RDKit CRN input generation does not support SynReactor-specific template strategies"
        )
    reaction = _reaction(rule)
    molecules = []
    for text in keys:
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            raise ValueError(f"Invalid substrate: {text}")
        molecules.append(Chem.AddHs(molecule))
    if len(molecules) != reaction.GetNumReactantTemplates():
        return index, keys, []
    from ..backend import record_operation

    record_operation("rdkit_cpp_rule_application")
    results = []
    seen = set()
    for assignment in permutations(range(len(molecules))):
        outcomes = reaction.RunReactants(
            tuple(molecules[i] for i in assignment), maxProducts=10000
        )
        if len(outcomes) >= 10000:
            raise RuntimeError("RDKit reaction product limit reached")
        for outcome in outcomes:
            try:
                product = _canonical_product(outcome, explicit_h)
            except (ValueError, RuntimeError):
                continue
            if product not in seen:
                seen.add(product)
                results.append(product)
    return index, keys, results
