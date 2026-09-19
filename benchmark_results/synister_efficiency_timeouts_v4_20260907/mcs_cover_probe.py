import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS


def molecule(matrix, elements):
    m = Chem.RWMol()
    for element in elements:
        m.AddAtom(Chem.Atom(int(element)))
    for i, j in zip(*np.triu_indices(len(elements), 1)):
        weight = matrix[i, j]
        if weight:
            m.AddBond(int(i), int(j), {1:Chem.BondType.SINGLE, 1.5:Chem.BondType.AROMATIC, 2:Chem.BondType.DOUBLE, 3:Chem.BondType.TRIPLE}[weight])
    result = m.GetMol()
    result.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(result)
    return result


def cover(a,b,er,ep):
    fixed={}
    sizes=[]
    for _ in range(16):
        ra=[i for i in range(len(er)) if i not in fixed]
        pa=[i for i in range(len(ep)) if i not in set(fixed.values())]
        ar=molecule(a[np.ix_(ra,ra)],[er[i] for i in ra])
        bp=molecule(b[np.ix_(pa,pa)],[ep[i] for i in pa])
        result=rdFMCS.FindMCS([ar,bp],timeout=1,ringMatchesRingOnly=True)
        if result.numAtoms<3:break
        query=Chem.MolFromSmarts(result.smartsString)
        am,bm=ar.GetSubstructMatch(query),bp.GetSubstructMatch(query)
        fixed.update({ra[i]:pa[j] for i,j in zip(am,bm)})
        sizes.append(len(am))
    return fixed,sizes
