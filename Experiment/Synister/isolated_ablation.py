"""Auditable, experiment-only source interventions for E4.

Compile the production function in a private namespace. Each removal requires
one exact AST anchor; the production module and its default API stay unchanged.
The transformed source is an explicit experimental solver, not a production
configuration. Recursive minimum proof uses that same private function.
"""

import ast
from hashlib import sha256
import inspect

from synkit.Chem.Mapper.exact import distance


VARIANTS = {
    'full': 'Unchanged production function compiled in the same private namespace.',
    'no_cross_assignment': 'Remove prefix cross-cost LAP contribution only; keep root/profile/conditioned assignment bounds and atom-order rules.',
    'no_residual_bound': 'Remove residual internal bond-mass lower contribution only; retain mass bookkeeping and numeric upper bound.',
    'no_prefix_profile': 'Remove non-tight prefix incident-profile bound only; keep root domain filtering, tight-face propagation, atom ordering and conditioned profiles.',
    'no_conditioned_filter': 'Remove conditioned candidate rejection only; retain conditioned bound and mask calculation for identical row-choice rules.',
    'no_symmetry': 'Coupled removal of symmetry discovery, candidate pruning, stabilizers and indexed expansion; retain full indexed output.',
}


class Removal(ast.NodeTransformer):
    def __init__(self, variant):
        self.variant = variant
        self.changed = 0

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return node
        name = node.targets[0].id
        if self.variant == 'no_cross_assignment' and name == 'cross_lower':
            if not isinstance(node.value, ast.Call) or ast.unparse(node.value.func) != '_blocked_assignment_extreme':
                raise ValueError('Cross-assignment anchor changed')
            node.value = ast.Constant(0.0)
            self.changed += 1
        elif self.variant == 'no_residual_bound' and name == 'lower' and ast.unparse(node.value) == 'cross_lower + internal_lower':
            node.value = ast.Name(id='cross_lower', ctx=ast.Load())
            self.changed += 1
        elif self.variant == 'no_conditioned_filter' and name == 'allowed_images':
            if ast.unparse(node.value) != 'set(remaining_images[branch_allowed[selected_position - depth]]) if branch_allowed is not None else None':
                raise ValueError('Conditioned-filter anchor changed')
            node.value = ast.Constant(None)
            self.changed += 1
        return node

    def visit_If(self, node):
        node = self.generic_visit(node)
        if self.variant == 'no_prefix_profile' and ast.unparse(node.test) == 'profile_pair_costs is not None and (not base_pruned) and (not tight_profile_rows)':
            node.test = ast.Constant(False)
            self.changed += 1
        return node


def compile_variant(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown isolated ablation')
    source = inspect.getsource(distance.enumerate_distance_mappings)
    tree = ast.parse(source)
    transformer = Removal(variant)
    tree = transformer.visit(tree)
    expected = 0 if variant in ('full', 'no_symmetry') else 1
    if transformer.changed != expected:
        raise ValueError(f'Expected {expected} intervention anchor, found {transformer.changed}')
    ast.fix_missing_locations(tree)
    transformed = ast.unparse(tree)+'\n'
    # Copying globals isolates recursive dispatch and observer wrappers.
    namespace = dict(vars(distance))
    filename = f'<synister-e4:{variant}>'
    exec(compile(transformed, filename, 'exec'), namespace)
    options = {'symmetry_pruning': variant != 'no_symmetry', 'expand_symmetry': variant != 'no_symmetry'}
    identity = {'variant': variant, 'description': VARIANTS[variant],
                'production_function_sha256': sha256(source.encode()).hexdigest(),
                'transformed_function_sha256': sha256(transformed.encode()).hexdigest(),
                'transformed_source': transformed, 'changed_anchors': transformer.changed,
                'filename': filename, 'options': options}
    return namespace['enumerate_distance_mappings'], identity
