from __future__ import annotations
import networkx as nx
from collections import Counter
from hashlib import blake2b
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import numpy as np
from synkit.IO import rsmi_to_graph


@dataclass
class WLRnxFps:
    """
    Weisfeiler-Lehman Reaction Fingerprint Sketch (NetworkX).

    :param radius: number of WL refinement iterations
    :type  radius: int
    :param size: bit budget for the parity sketch
    :type  size: int
    :param to_array: whether to return fingerprint as NumPy array
    :type  to_array: bool

    Usage example:
    >>> rsmi = ('COc1cc(NC(N)=S)ccc1-n1cnc(C)c1.O=C1C(Br)CCCC1c1ccc(Cl)cc1Cl'
    >>> +'>>Br.COc1cc(Nc2nc3c(s2)CCCC3c2ccc(Cl)cc2Cl)ccc1-n1cnc(C)c1.O')
    >>> react, prod = rsmi_to_graph(rsmi, drop_non_aam=False, use_index_as_atom_map=False)
    >>> fps = WLRnxFps(radius=2, size=1024, to_array=False).fit(react, prod)
    >>> bits = fps.fingerprint
    """

    radius: int = 2
    size: int = 1024
    to_array: bool = False

    _tokens_R: Optional[Counter] = field(init=False, default=None)
    _tokens_P: Optional[Counter] = field(init=False, default=None)
    _delta: Optional[Counter] = field(init=False, default=None)
    _support: Optional[List[int]] = field(init=False, default=None)
    _fingerprint: Optional[Union[List[int], np.ndarray]] = field(
        init=False, default=None
    )

    def fit(self, react: nx.Graph, prod: nx.Graph) -> WLRnxFps:
        """
        Compute WL tokens for reactant and product graphs, then build parity sketch on Δ-support.

        :param react: reactant graph with node attrs 'element','aromatic','hcount','charge'
        :type react: nx.Graph
        :param prod: product graph with same node/edge attrs
        :type prod: nx.Graph
        :returns: self
        :rtype: WLRnxFps
        :raises ValueError: if size is not positive
        """
        if self.size <= 0:
            raise ValueError("size must be a positive integer")

        def wl_tokens(G: nx.Graph) -> Counter:
            labels: Dict[int, int] = {}
            for n, attrs in G.nodes(data=True):
                atom_tuple = (
                    attrs.get("element"),
                    bool(attrs.get("aromatic", False)),
                    int(attrs.get("charge", 0)),
                    int(attrs.get("hcount", 0)),
                    G.degree(n),
                )
                labels[n] = _h64(("init", atom_tuple))
            cnt = Counter(labels.values())
            for k in range(1, self.radius + 1):
                new_labels: Dict[int, int] = {}
                for n in G.nodes():
                    neigh = []
                    for m in G.neighbors(n):
                        bond_order = float(G.edges[n, m].get("order", 1.0))
                        neigh.append((_h64(("bond", bond_order)), labels[m]))
                    neigh.sort()
                    new_labels[n] = _h64(("wl", k, labels[n], tuple(neigh)))
                labels = new_labels
                cnt.update(labels.values())
            return cnt

        TR = wl_tokens(react)
        TP = wl_tokens(prod)

        Delta = Counter(TP)
        for h, v in TR.items():
            Delta[h] -= v
            if Delta[h] == 0:
                del Delta[h]

        support = list(Delta.keys())
        self._tokens_R = TR
        self._tokens_P = TP
        self._delta = Delta
        self._support = support

        bits = np.zeros(self.size, dtype=int) if self.to_array else [0] * self.size
        for h in support:
            idx = h % self.size
            if self.to_array:
                bits[idx] ^= 1
            else:
                bits[idx] = bits[idx] ^ 1
        self._fingerprint = bits

        return self

    @classmethod
    def from_rsmi(
        cls,
        rsmi: str,
        radius: int = 2,
        size: int = 1024,
        to_array: bool = False,
        drop_non_aam: bool = False,
        use_index_as_atom_map: bool = False,
    ) -> WLRnxFps:
        """
        Build WLRnxFps directly from a reaction SMILES string.

        :param rsmi: reaction SMILES string
        :type rsmi: str
        :param radius: number of WL refinement iterations
        :type radius: int
        :param size: bit budget for the parity sketch
        :type size: int
        :param to_array: return fingerprint as NumPy array if True
        :type to_array: bool
        :param drop_non_aam: drop atoms without atom-atom mapping
        :type drop_non_aam: bool
        :param use_index_as_atom_map: interpret node indices as atom map numbers
        :type use_index_as_atom_map: bool
        :returns: fitted WLRnxFps instance
        :rtype: WLRnxFps
        :raises ValueError: on invalid SMILES parsing
        """
        try:
            react, prod = rsmi_to_graph(
                rsmi,
                drop_non_aam=drop_non_aam,
                use_index_as_atom_map=use_index_as_atom_map,
            )
        except Exception as e:
            raise ValueError(f"Failed to parse rsmi: {e}")

        return cls(radius=radius, size=size, to_array=to_array).fit(react, prod)

    @property
    def tokens_R(self) -> Counter:
        """
        :returns: WL token counts for reactant
        :rtype: Counter
        :raises AttributeError: if fit() has not been called
        """
        if self._tokens_R is None:
            raise AttributeError("Call fit() before accessing tokens_R")
        return self._tokens_R

    @property
    def tokens_P(self) -> Counter:
        """
        :returns: WL token counts for product
        :rtype: Counter
        :raises AttributeError: if fit() has not been called
        """
        if self._tokens_P is None:
            raise AttributeError("Call fit() before accessing tokens_P")
        return self._tokens_P

    @property
    def delta(self) -> Counter:
        """
        :returns: Signed token difference (product - reactant)
        :rtype: Counter
        :raises AttributeError: if fit() has not been called
        """
        if self._delta is None:
            raise AttributeError("Call fit() before accessing delta")
        return self._delta

    @property
    def support(self) -> List[int]:
        """
        :returns: Tokens with non-zero delta
        :rtype: List[int]
        :raises AttributeError: if fit() has not been called
        """
        if self._support is None:
            raise AttributeError("Call fit() before accessing support")
        return self._support

    @property
    def fingerprint(self) -> Union[List[int], np.ndarray]:
        """
        :returns: Parity sketch bit vector (0/1)
        :rtype: Union[List[int], numpy.ndarray]
        :raises AttributeError: if fit() has not been called
        """
        if self._fingerprint is None:
            raise AttributeError("Call fit() before accessing fingerprint")
        return self._fingerprint

    def __repr__(self) -> str:
        support_len = len(self._support) if self._support is not None else 0
        return (
            f"<WLRnxFps radius={self.radius} size={self.size} "
            f"support={support_len} to_array={self.to_array}>"
        )

    def help(self) -> None:
        """
        Print usage examples and class docstring.

        :returns: None
        """
        print(self.__doc__)


def _h64(obj: Any) -> int:
    """
    Compute a stable 64-bit hash of an object.

    :param obj: any hashable representation
    :type obj: Any
    :returns: 64-bit integer hash
    :rtype: int
    """
    h = blake2b(digest_size=8)
    h.update(repr(obj).encode("utf-8"))
    return int.from_bytes(h.digest(), "little")
