import pytest

from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings
from synkit.Chem.Mapper.exact.edit_support import enumerate_binary_edit_support_mappings
from synkit.Chem.Mapper.identifiability import Endpoint


@pytest.mark.parametrize('backend', ['assignment', 'edit_support'])
def test_python_callback_exception_is_propagated_not_reported_complete(backend):
    endpoint = Endpoint((6, 6), (0, 0), (0, 0), ())
    observed = []
    failure = RuntimeError('deliberate consumer interruption')

    def consume(mapping, cost):
        observed.append((tuple(mapping), cost))
        raise failure

    graphs = [endpoint.graph(), endpoint.graph()]
    with pytest.raises(RuntimeError) as caught:
        if backend == 'assignment':
            enumerate_distance_mappings(graphs, CD=0, compute_minimum_cost=False,
                                        collect_mappings=False, mapping_callback=consume)
        else:
            enumerate_binary_edit_support_mappings(graphs, CD=0,
                                                   collect_mappings=False, mapping_callback=consume)
    assert caught.value is failure and len(observed) == 1
    assert observed[0][1] == 0
