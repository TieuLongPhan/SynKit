from copy import deepcopy
import json

import pytest

from pathlib import Path

TRACE = Path(__file__).resolve().parents[3]/"paper/synister/evidence/production_trace_v1.json"
from Experiment.Synister.trace_exact_search import audit, literal_maps, run
from synkit.Chem.Mapper.identifiability import parse_reaction


def test_actual_minimum_assignment_event_and_literal_subtree():
    trace = run()
    assert trace['summary']['visited_nodes'] == 191
    assert trace['summary']['traced_untraced_equal']
    assert len(trace['summary']['returned_maps']) == 8
    node = trace['nodes'][trace['highlight_node_ids'][0]]
    assert node['committed_cost'] == 3 and node['incumbent_or_target'] == 6
    assert node['decision'] == 'lower_reject'
    assert node['first_decisive_bound'] == 'assignment_plus_residual'
    assert node['bound']['remaining_lower'] == 5
    assert not node['bound']['profile_evaluated'] and not node['bound']['conditioned_evaluated']
    assert node['literal_subtree'] == {'completions': 720, 'minimum_cd': 10, 'maximum_cd': 24}
    carbon = next(b for b in node['bound']['blocks'] if b['element'] == 6)
    assert carbon['independent_row_minima'] == 1 and carbon['literal_assignment_minimum'] == 2
    assert all(n['phase'] == 'collected_minimum' and not n['strict_improvement'] for n in trace['nodes'])


def test_supplied_and_collected_modes_have_separate_traversals():
    trace = run(6)
    assert not trace['highlight_node_ids']
    assert trace['summary']['visited_nodes'] == 33
    assert len(trace['summary']['returned_maps']) == 8
    assert all(n['phase'] == 'supplied_cd' for n in trace['nodes'])
    assert trace['candidate_rejections']
    for event in trace['candidate_rejections']:
        assert event['literal_minimum_cd'] > event['limit']


def test_empty_and_capped_outputs_remain_distinct():
    empty, capped = run(7), run(6, max_mappings=1)
    assert empty['summary']['complete'] and not empty['summary']['returned_maps']
    assert not capped['summary']['complete'] and len(capped['summary']['returned_maps']) == 1
    assert capped['summary']['truncation_reason'] == 'mapping_limit'
    assert capped['nodes'][0]['decision'] == 'interrupted'


def test_invalid_observed_bound_is_rejected_by_literal_audit():
    from types import SimpleNamespace
    data = run()
    corrupted = deepcopy(data["nodes"][data["highlight_node_ids"][0]])
    corrupted['bound']['remaining_lower'] = 100
    r, p = parse_reaction('>>'.join(data['unmapped_smiles']))
    with pytest.raises(ValueError, match='bound exceeds'):
        audit(SimpleNamespace(nodes=[corrupted], candidate_rejections=[]), literal_maps(r, p))


def test_archived_larger_space_and_all_trace_source_identities():
    data = json.loads(TRACE.read_text())
    larger = data['larger_compatible_space']
    assert larger['case_id'] == 'reaction_058'
    assert larger['summary']['literal_maps'] == 86400
    assert larger['summary']['visited_nodes'] == 273
    assert larger['summary']['minimum_cd'] == 4
    assert len(larger['summary']['returned_maps']) == 2
    assert json.loads(TRACE.read_text()) == data
