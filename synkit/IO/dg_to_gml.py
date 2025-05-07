import regex
import importlib.util
from synkit.IO.debug import setup_logging
from synkit.Chem.Reaction.standardize import Standardize

logger = setup_logging()

if importlib.util.find_spec("mod"):
    from mod import DGVertexMapper, smiles, Rule
else:
    DGVertexMapper = None
    smiles = None
    Rule = None
    logger.warning("Optional 'mod' package not found: GML conversion will be limited.")


class DGToGML:
    def __init__(self) -> None:
        self.standardizer = Standardize()
        pass

    @staticmethod
    def getReactionSmiles(dg):
        origSmiles = {}
        for v in dg.vertices:
            s = v.graph.smilesWithIds
            s = regex.sub(":([0-9]+)]", ":o\\1]", s)
            origSmiles[v.graph] = s

        res = {}
        for e in dg.edges:
            # vms = DGVertexMapper(e, rightLimit=1, leftLimit=1)
            vms = DGVertexMapper(e, rightLimit=1)
            # vms = DGVertexMapper(e)
            eductSmiles = [origSmiles[g] for g in vms.left]

            for ev in vms.left.vertices:
                s = eductSmiles[ev.graphIndex]
                s = s.replace(f":o{ev.vertex.id}]", f":{ev.id}]")
                eductSmiles[ev.graphIndex] = s

            strs = set()
            # for vm in DGVertexMapper(e, rightLimit=1, leftLimit=1):
            for vm in DGVertexMapper(e, rightLimit=1):
                # for vm in DGVertexMapper(e):
                productSmiles = [origSmiles[g] for g in vms.right]
                for ev in vms.left.vertices:
                    pv = vm.map[ev]
                    if not pv:
                        continue
                    s = productSmiles[pv.graphIndex]
                    s = s.replace(f":o{pv.vertex.id}]", f":{ev.id}]")
                    productSmiles[pv.graphIndex] = s
                count = vms.left.numVertices
                for pv in vms.right.vertices:
                    ev = vm.map.inverse(pv)
                    if ev:
                        continue
                    s = productSmiles[pv.graphIndex]
                    s = s.replace(f":o{pv.vertex.id}]", f":{count}]")
                    count += 1
                    productSmiles[pv.graphIndex] = s
                left = ".".join(eductSmiles)
                right = ".".join(productSmiles)
                s = f"{left}>>{right}"
                assert ":o" not in s
                strs.add(s)
            res[e] = list(sorted(strs))
        return res

    @staticmethod
    def parseReactionSmiles(line: str) -> Rule:
        sLeft, sRight = line.split(">>")
        ssLeft = sLeft.split(".")
        ssRight = sRight.split(".")
        mLeft = [smiles(s, add=False) for s in ssLeft]
        mRight = [smiles(s, add=False) for s in ssRight]

        def printGraph(g):
            extFromInt = {}
            for iExt in range(g.minExternalId, g.maxExternalId + 1):
                v = g.getVertexFromExternalId(iExt)
                if not v.isNull():
                    extFromInt[v] = iExt
            s = ""
            for v in g.vertices:
                assert v in extFromInt
                s += '\t\tnode [ id %d label "%s" ]\n' % (extFromInt[v], v.stringLabel)
            for e in g.edges:
                s += '\t\tedge [ source %d target %d label "%s" ]\n' % (
                    extFromInt[e.source],
                    extFromInt[e.target],
                    e.stringLabel,
                )
            return s

        s = "rule [\n\tleft [\n"
        for m in mLeft:
            s += printGraph(m)
        s += "\t]\n\tright [\n"
        for m in mRight:
            s += printGraph(m)
        s += "\t]\n]\n"
        return s, Rule.fromGMLString(s, add=False)

    def fit(self, dg, origSmiles):
        """
        Matches the original SMILES to a list of generated reaction SMILES and
        returns the parsed reaction.

        Parameters:
        - dg (DataGenerator): The data generator instance containing the reactions.
        - origSmiles (str): The original SMILES string to match.

        Returns:
        - Parsed reaction if a match is found; otherwise, None.
        """
        try:
            res = DGToGML.getReactionSmiles(dg)
            smiles_list = [value for values in res.values() for value in values]

            smiles_standard = [
                self.standardizer.fit(rsmi, True, True) for rsmi in smiles_list
            ]
            origSmiles_standard = self.standardizer.fit(origSmiles, True, True)

            for index, value in enumerate(smiles_standard):
                if value == origSmiles_standard:
                    return self.parseReactionSmiles(smiles_list[index])

            return None
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return None
