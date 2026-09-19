"""Independent SciPy assignment checks across the diagnostic's large-cost domain."""
import ctypes,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
R=Path(__file__).resolve().parent
lib=ctypes.CDLL((R/"guard_library.txt").read_text().strip())
function=lib.synkit_assignment_seed_certificate
pointer=ctypes.POINTER(ctypes.c_int32)
function.argtypes=[ctypes.c_int]+[pointer]*7
inf=100000000
rng=np.random.default_rng(20260909)
checked=forced_checks=0
largest=0
for trial in range(200):
    n=(16,32,64)[trial%3]
    matrix=rng.integers(0,1000001,size=(n,n),dtype=np.int32)
    if trial%4==0:
        matrix=rng.integers(900000,1000001,size=(n,n),dtype=np.int32)
    if trial%4 in (1,2):
        matrix[rng.random((n,n))<0.96]=inf
        mapping=rng.permutation(n)
        matrix[np.arange(n),mapping]=rng.integers(900000,1000001,size=n,dtype=np.int32)
    real=matrix.astype(float)
    real[real==inf]=np.inf
    rows,cols=linear_sum_assignment(real)
    expected=int(real[rows,cols].sum())
    match,u,v=[np.empty(n,dtype=np.int32) for _ in range(3)]
    forced=np.empty((n,n),dtype=np.int32)
    seed_v=rng.integers(-2000000000,2000000001,n,dtype=np.int32)
    seed_match=rng.integers(-1,n,n,dtype=np.int32)
    observed=function(n,*(x.ctypes.data_as(pointer) for x in (matrix,seed_v,seed_match,match,u,v,forced)))
    assert observed==expected,(trial,observed,expected)
    assert int(u.sum()+v.sum())==expected
    assert np.all(u[:,None]+v[None,:]<=matrix)
    for i,j in rng.integers(0,n,size=(32,2)):
        expected_forced=inf
        if matrix[i,j]!=inf:
            sub=np.delete(np.delete(real,i,axis=0),j,axis=1)
            try:
                a,b=linear_sum_assignment(sub)
                expected_forced=int(real[i,j]+sub[a,b].sum())
            except ValueError:
                pass
        assert forced[i,j]==expected_forced,(trial,int(i),int(j),int(forced[i,j]),expected_forced)
        forced_checks+=1
    checked+=1
    largest=max(largest,expected)
summary={"matrices_checked":checked,"forced_edges_checked":forced_checks,"largest_optimum":largest,"failures":0,"seed":20260909,"oracle":"SciPy linear_sum_assignment; each sampled forced edge solved independently after row/column deletion"}
(R/"assignment_stress.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary))
