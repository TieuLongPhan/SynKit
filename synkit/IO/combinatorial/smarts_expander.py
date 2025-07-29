import re
import itertools
from typing import List, Dict, Tuple, Iterator, Union


class SMARTSExpander:
    """
    Efficiently enumerate all valid reaction SMARTS by expanding atom-list
    placeholders like [C,N,O,P,S:9], ensuring that each atom-map uses the same
    element everywhere it appears (on both sides of a reaction).

    :param smarts: SMARTS string, possibly containing one or more atom-list placeholders.
    :type smarts: str
    :returns: Expanded SMARTS strings without atom-list placeholders.
    :rtype: List[str]
    :raises ValueError: If no valid expansions exist due to incompatible element lists.

    Example usage::

        >>> rxn = (
        ...     '[H+:6].[C:7](-[O:8](-[H:12]))(-[C,N,O,P,S:9])'
        ...     '(-[C,N,O,P,S:10])(-[H:11]).'
        ...     '[C:2](-[S:4](-[C,N,O,P,S:5]))(-[C,N,O,P,S:1])(=[O:3])>>'
        ...     '[S:4](-[H:6])(-[C,N,O,P,S:5]).[H+:12].'
        ...     '[C:7](-[O:8](-[C:2](-[C,N,O,P,S:1])(=[O:3])))(-[C,N,O,P,S:9])'
        ...     '(-[C,N,O,P,S:10])(-[H:11])'
        ... )
        >>> ex = SMARTSExpander.expand(rxn)
        >>> len(ex)
        625
        >>> ex[:3]  # first three expansions
        ['[H+:6].[C:7](-[O:8](-[H:12]))(-[C:9])(-[C:10])(-[H:11]).[C:2]...'
        '>>',
        '...',
        '...']
    """

    _PAT = re.compile(r"\[([A-Z][a-z]?(?:,[A-Z][a-z]?)*)(:[0-9]+)\]")

    @staticmethod
    def _extract_map_to_elements(matches: List[re.Match]) -> Dict[str, List[str]]:
        """
        Build a mapping from atom-map to the intersection of allowed elements.

        :param matches: List of regex match objects for placeholders.
        :type matches: List[re.Match]
        :returns: Dictionary mapping ":map" to sorted list of shared elements.
        :rtype: Dict[str, List[str]]
        """
        amap2set: Dict[str, set] = {}
        for m in matches:
            elems = set(m.group(1).split(","))
            amap = m.group(2)
            if amap not in amap2set:
                amap2set[amap] = elems
            else:
                amap2set[amap] &= elems
        for amap, s in amap2set.items():
            if not s:
                raise ValueError(f"No overlapping elements for atom-map {amap}")
        return {amap: sorted(s) for amap, s in amap2set.items()}

    @staticmethod
    def _build_template(
        smarts: str, matches: List[re.Match]
    ) -> Tuple[List[Union[str, str]], List[str]]:
        """
        Build a list of string segments and placeholders for reconstruction.

        :param smarts: Original SMARTS string.
        :type smarts: str
        :param matches: Regex matches for placeholders.
        :type matches: List[re.Match]
        :returns: Tuple of list of segments and placeholder order.
        :rtype: Tuple[List[Union[str, str]], List[str]]
        """
        segments: List[Union[str, str]] = []
        placeholder_order: List[str] = []
        last = 0
        for m in matches:
            # fmt: off
            segments.append(smarts[last: m.start()])
            # fmt: on
            amap = m.group(2)
            segments.append(amap)
            placeholder_order.append(amap)
            last = m.end()
        segments.append(smarts[last:])
        return segments, placeholder_order

    @classmethod
    def expand_iter(cls, smarts: str) -> Iterator[str]:
        """
        Yield expanded SMARTS strings lazily.

        :param smarts: SMARTS string with placeholders.
        :type smarts: str
        :yields: One expanded SMARTS string at a time.
        :rtype: Iterator[str]

        :raises ValueError: If no valid expansions due to incompatible lists.
        """
        matches = list(cls._PAT.finditer(smarts))
        if not matches:
            yield smarts
            return

        amap2els = cls._extract_map_to_elements(matches)
        segments, order = cls._build_template(smarts, matches)
        unique_maps = list(dict.fromkeys(order))

        pools = [amap2els[am] for am in unique_maps]
        for combo in itertools.product(*pools):
            mapping = dict(zip(unique_maps, combo))
            out = []
            for seg in segments:
                if seg in mapping:
                    out.append(f"[{mapping[seg]}{seg}]")
                else:
                    out.append(seg)
            yield "".join(out)

    @classmethod
    def expand(cls, smarts: str) -> List[str]:
        """
        Return a list of all expanded SMARTS.

        :param smarts: SMARTS string with placeholders.
        :type smarts: str
        :returns: List of expanded SMARTS strings.
        :rtype: List[str]

        :raises ValueError: If no valid expansions exist.
        """
        return list(cls.expand_iter(smarts))


# # --- Example usage ---

# if __name__ == "__main__":
#     rxn = (
#         '[H+:6].[C:7](-[O:8](-[H:12]))(-[C,N,O,P,S:9])(-[C,N,O,P,S:10])(-[H:11]).'
#         '[C:2](-[S:4](-[C,N,O,P,S:5]))(-[C,N,O,P,S:1])(=[O:3])>>'
#         '[S:4](-[H:6])(-[C,N,O,P,S:5]).[H+:12].'
#         '[C:7](-[O:8](-[C:2](-[C,N,O,P,S:1])(=[O:3])))(-[C,N,O,P,S:9])(-[C,N,O,P,S:10])(-[H:11])'
#     )
#     n = 0
#     for i, s in enumerate(SMARTSExpander.expand_iter(rxn)):
#         if i < 3 or i > 621:
#             print(f"{i+1}: {s}")
#         n += 1
#     print(f"Total: {n} enumerated SMARTS.")
