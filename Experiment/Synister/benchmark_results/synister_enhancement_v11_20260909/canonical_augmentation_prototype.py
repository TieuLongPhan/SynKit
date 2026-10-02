"""Small explicit-group oracle for canonical augmentation; not a production backend."""
from itertools import permutations,combinations
from collections import Counter
from pathlib import Path
import json,time
import numpy as np
import networkx as nx

def run(a,b,labels):
    n=len(a)
    def group(matrix):
        return [p for p in permutations(range(n))
                if all(labels[i]==labels[p[i]] for i in range(n))
                and np.array_equal(matrix,matrix[np.ix_(p,p)])]
    rg,pg=group(a),group(b)
    action=[(r,p) for r in rg for p in pg]
    def moved(edges,r,p):
        return tuple(sorted((r[i],p[j]) for i,j in edges))
    def canonical(edges):
        return min(moved(edges,r,p) for r,p in action)
    def marked(edges,edge):
        return min((moved(edges,r,p),(r[edge[0]],p[edge[1]])) for r,p in action)
    levels=[{()}]
    for depth in range(n):
        children=[]
        for parent in levels[-1]:
            stabilizer=[(r,p) for r,p in action if moved(parent,r,p)==parent]
            rows={i for i,j in parent}; cols={j for i,j in parent}
            extensions={(i,j) for i in range(n) for j in range(n)
                        if i not in rows and j not in cols and labels[i]==labels[j]}
            while extensions:
                edge=min(extensions)
                extensions.difference_update((r[edge[0]],p[edge[1]]) for r,p in stabilizer)
                child=tuple(sorted((*parent,edge)))
                if marked(child,edge)==min(marked(child,e) for e in child):
                    children.append(canonical(child))
        assert len(children)==len(set(children)), ("duplicate construction",depth)
        levels.append(set(children))
    # Independent exhaustive partial-bijection quotient at EVERY depth.
    for depth,observed in enumerate(levels):
        expected=set()
        for rows in combinations(range(n),depth):
            for cols in permutations(range(n),depth):
                if all(labels[i]==labels[j] for i,j in zip(rows,cols)):
                    expected.add(canonical(tuple(zip(rows,cols))))
        assert observed==expected,("missing orbit",depth,len(observed),len(expected))
    costs=Counter()
    for edges in levels[-1]:
        m=dict(edges)
        costs[float(sum(abs(a[i,j]-b[m[i],m[j]]) for i in range(n) for j in range(i)))]+=1
    return {"n":n,"reactant_order":len(rg),"product_order":len(pg),
            "partial_orbit_counts":[len(x) for x in levels],"full_orbits_by_cost":dict(costs)}

if __name__=="__main__":
    start=time.perf_counter()
    cases=[]
    for ga,gb,labels in [
        (nx.empty_graph(4),nx.empty_graph(4),[6]*4),
        (nx.star_graph(3),nx.path_graph(4),[6]*4),
        (nx.cycle_graph(5),nx.path_graph(5),[6]*5),
        (nx.path_graph(4),nx.cycle_graph(4),[6,8,6,8]),
    ]:
        cases.append(run(nx.to_numpy_array(ga),nx.to_numpy_array(gb),labels))
    result={"status":"prototype equivalence passed","cases":cases,
            "elapsed_seconds":time.perf_counter()-start,
            "production_integration":False,"distance_pruning":False}
    Path(__file__).with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)
