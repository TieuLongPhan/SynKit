from Experiment.Synister.audit_two_sided_its import audit


def test_literal_two_sided_action_equals_full_attributed_isomorphism():
    report = audit()
    assert report['all_passed'] and len(report['cases']) == 10
    assert report['comparisons'] == 1855
    rows = {r['case_id']: r for r in report['cases']}
    empty = rows['empty_three']
    assert empty['reactant_group_order'] * empty['product_group_order'] > empty['maps']
    assert empty['two_sided_classes'] == 1
    one_sided = rows['product_only_insufficient']
    assert one_sided['product_orbits'] == 3 and one_sided['two_sided_classes'] == 1
    unary = rows['paired_unary_attributes']
    assert unary['reactant_group_order'] == unary['product_group_order'] == 1
    assert unary['two_sided_classes'] == 2
