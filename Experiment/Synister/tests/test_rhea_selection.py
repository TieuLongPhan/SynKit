import pytest
from Experiment.Synister.select_rhea import choose, supported


def test_numeric_master_deduplication_and_input_order_invariance():
    rows=[{'master_id':str(i),'endpoint_sha256':str(i)} for i in range(20)]
    rows += [{'master_id':'100','endpoint_sha256':'2'}]
    a,frame=choose(rows,10)
    assert (a,frame)==choose(list(reversed(rows)),10)
    assert len(frame)==20 and '100' not in {r['master_id'] for r in frame}
    with pytest.raises(ValueError,match='amendment'): choose(rows,21)


def test_rhea_all_component_domain_does_not_drop_protons():
    with pytest.raises(ValueError): supported('CC.[H+]>>CC.[H+]')
    with pytest.raises(ValueError): supported('*C>>*C')
    value,key,n=supported('[H]OC>>CO')
    assert n==2 and key==supported('CO>>CO')[1]
