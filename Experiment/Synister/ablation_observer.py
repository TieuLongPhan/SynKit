"""Separate instrumented E4 counters and inclusive operation timings.

Trace overhead is intentionally excluded from ordinary performance runs.
Operation timers can overlap and are never added into a causal time partition.
"""

from collections import Counter, defaultdict
import sys
import time


class Observer:
    def __init__(self, function, identity):
        self.filename = identity['filename']
        lines = identity['transformed_source'].splitlines()
        anchors = {
            'committed_rejections': 'pruned_branches += 1',
            'lower_rejections': 'lower_bound_pruned_branches += 1',
            'upper_rejections': 'upper_bound_pruned_branches += 1',
            'symmetry_rejections': 'symmetry_pruned_branches += 1',
            'tight_profile_rejections': 'tight_profile_pruned += 1',
            'candidates': 'for product_atom in candidate_images:',
            'output': 'for expansion_index, selected_mapping in enumerate(expanded):',
        }
        self.anchors = {}
        for name, text in anchors.items():
            found = [i+1 for i,line in enumerate(lines) if line.strip() == text]
            if len(found) != 1:
                raise ValueError('Observer anchor changed: '+name)
            self.anchors[found[0]] = name
        namespace = function.__globals__
        self.operations = {}
        for name in ('_blocked_assignment_extreme', '_conditioned_profile_bound', '_assignment_edge_lower_bounds',
                     'bounded_automorphism_permutations', 'largest_cyclic_subgroup',
                     'point_stabilizer_generators_checked', 'orbital_candidate_witnesses'):
            self.operations[namespace[name].__code__] = name
        self.phases = []
        self.calls = {}
        self.nodes = {}
        self.counters = defaultdict(Counter)
        self.timings = defaultdict(Counter)
        self.production_statistics = {}

    def profile(self, frame, event, arg):
        if event not in ('call', 'return'):
            return
        if frame.f_code.co_filename == self.filename and frame.f_code.co_name == 'enumerate_distance_mappings':
            if event == 'call':
                self.phases.append('minimum_proof' if frame.f_locals['_optimization_only'] else 'enumeration')
            else:
                if arg is not None and hasattr(arg, 'backend_statistics'):
                    self.production_statistics[self.phases[-1]] = arg.backend_statistics
                self.phases.pop()
        operation = self.operations.get(frame.f_code)
        if operation and self.phases:
            if event == 'call':
                phase = self.phases[-1]
                self.calls[id(frame)] = phase, operation, time.perf_counter()
                self.counters[phase][operation+'_calls'] += 1
            else:
                phase, operation, started = self.calls.pop(id(frame))
                self.timings[phase][operation+'_inclusive_seconds'] += time.perf_counter()-started

    def trace(self, frame, event, arg):
        if frame.f_code.co_filename != self.filename:
            return None
        name, local = frame.f_code.co_name, frame.f_locals
        if name not in ('visit', 'remaining_assignment_interval'):
            return None
        phase = 'minimum_proof' if local.get('_optimization_only', local.get('strict_improvement', False)) else 'enumeration'
        # The profile hook maintains the actual recursive proof phase, including
        # non-half-integer controls where strict_improvement can be false.
        if self.phases:
            phase = self.phases[-1]
        counts = self.counters[phase]
        if name == 'remaining_assignment_interval':
            if event == 'return' and id(frame.f_back) in self.nodes:
                node = self.nodes[id(frame.f_back)]
                limit = float(local['limit'])
                def decisive(value):
                    return value > limit+local['tolerance'] or (local['strict_improvement'] and value >= limit)
                node['first_bound'] = ('cross_plus_residual_or_tight_root' if local['base_pruned'] else
                    'incident_profile' if 'profile_lower' in local and decisive(local['profile_lower']) else
                    'conditioned_profile' if decisive(local['committed_cost']+arg[0]) else None)
            return self.trace
        if event == 'call':
            counts['visited_nodes'] += 1
            self.nodes[id(frame)] = {'candidates_seen': False, 'output_started': None, 'phase': phase}
        elif id(frame) in self.nodes:
            node = self.nodes[id(frame)]
            action = self.anchors.get(frame.f_lineno) if event == 'line' else None
            if action and action.endswith('rejections'):
                counts[action] += 1
                if action == 'lower_rejections':
                    counts['first_decisive_'+str(node.get('first_bound'))] += 1
            elif action == 'candidates' and not node['candidates_seen']:
                node['candidates_seen'] = True
                atom = local['reactant_atom']
                for image in local['domains'][atom]:
                    if local['used_products'][image]:
                        continue
                    if local['allowed_images'] is not None and image not in local['allowed_images']:
                        counts['conditioned_domain_removals'] += 1
                    elif local['profile_compatible'] is not None and not local['profile_compatible'][atom,image]:
                        counts['tight_profile_domain_removals'] += 1
                counts['candidate_images_after_filters'] += len(local['candidate_images'])
            elif action == 'output' and node['output_started'] is None:
                node['output_started'] = time.perf_counter()
                counts['selected_representative_leaves'] += 1
            elif event == 'return':
                if node['output_started'] is not None:
                    self.timings[phase]['indexed_expansion_and_callback_seconds'] += time.perf_counter()-node['output_started']
                del self.nodes[id(frame)]
        return self.trace

    def __enter__(self):
        self.previous_trace, self.previous_profile = sys.gettrace(), sys.getprofile()
        sys.setprofile(self.profile)
        sys.settrace(self.trace)
        return self

    def __exit__(self, *exc):
        sys.settrace(self.previous_trace)
        sys.setprofile(self.previous_profile)

    def result(self):
        return {'scope': 'Separate instrumented run; inclusive timers overlap and include observer overhead. Rejections are ordered, not independently causal.',
                'production_statistics_by_phase': self.production_statistics,
                'phases': {phase: {'counters': dict(self.counters[phase]), 'timings': dict(self.timings[phase])}
                           for phase in sorted(self.counters)}}
