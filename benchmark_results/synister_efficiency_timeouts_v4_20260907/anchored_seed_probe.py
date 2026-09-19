"""Experimental typed quadratic-overlap seed relaxation."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix


def improve(a,b,er,ep,mapping,iterations=40,fixed=()):
    n=len(mapping)
    blocks=[(np.asarray([i for i in range(n) if er[i]==e and i not in fixed],dtype=int),np.asarray([j for j in range(n) if ep[j]==e and j not in set(fixed.values())],dtype=int)) for e in sorted(set(er))]
    blocks=[(rows,cols) for rows,cols in blocks if len(rows)]
    layers=[]
    for sign in (1,-1):
        values=np.unique(np.concatenate(((sign*a).ravel(),(sign*b).ravel())))
        previous=0.
        for value in values[values>0]:
            ar=csr_matrix((sign*a>=value).astype(float));bp=csr_matrix((sign*b>=value).astype(float))
            layers.append((float(value-previous),ar,bp,ar.T.tocsr(),bp.T.tocsr()))
            previous=value
    def gradient(p):
        g=np.zeros_like(p)
        for weight,ar,bp,at,bt in layers:
            g+=weight*((bp@(ar@p).T).T+(bt@(at@p).T).T)
        return g
    def project(p):
        result=np.asarray(mapping).copy()
        for rows,cols in blocks:
            i,j=linear_sum_assignment(-p[np.ix_(rows,cols)])
            result[rows[i]]=cols[j]
        return result
    def cost(m):return .5*float(np.abs(a-b[np.ix_(m,m)]).sum())
    best=np.asarray(mapping).copy();best_cost=cost(best)
    identity=np.zeros((n,n));identity[np.arange(n),best]=1
    uniform=np.zeros((n,n))
    for i,j in fixed.items():uniform[i,j]=1
    for rows,cols in blocks:uniform[np.ix_(rows,cols)]=1/len(cols)
    count=0
    for mixing in (.95,.7,.3,0.):
        p=mixing*identity+(1-mixing)*uniform
        for _ in range(iterations):
            g=gradient(p);candidate=project(g);candidate_cost=cost(candidate)
            if candidate_cost<best_cost:best,best_cost=candidate.copy(),candidate_cost
            q=np.zeros_like(p);q[np.arange(n),candidate]=1
            d=q-p;slope=float((g*d).sum());curve=.5*float((d*gradient(d)).sum())
            t=min(1.,max(0.,-slope/(2*curve))) if curve<0 else float(slope+curve>0)
            count+=1
            if t<1e-8 or np.linalg.norm(d)<1e-8:break
            p+=t*d
        candidate=project(p);candidate_cost=cost(candidate)
        if candidate_cost<best_cost:best,best_cost=candidate.copy(),candidate_cost
    return best.tolist(),{'cost':best_cost,'iterations':count}
