"""Exact weighted accumulation of reactant/product double orbits."""

from collections import Counter

import numpy as np

from ..spectrum import (
    ExactStructureSpectrumAccumulator,
    _attributed_its_graph,
    _changed_atoms_and_bonds,
    _template_context,
)
from .native_canonical import (
    PackedCanonicalKey,
    compact_code_identifier,
    native_canonical_code,
)
from .native_its import NativeITSCanonicalizer


def coordinate_orbits(n, generators, pairs=False):
    coordinates = (
        [(i, j) for i in range(n) for j in range(i + 1, n)] if pairs else list(range(n))
    )
    position = {value: i for i, value in enumerate(coordinates)}
    parent = list(range(len(coordinates)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for generator in generators:
        for i, value in enumerate(coordinates):
            image = (
                tuple(sorted((generator[value[0]], generator[value[1]])))
                if pairs
                else generator[value]
            )
            left, right = find(i), find(position[image])
            if left != right:
                parent[right] = left
    groups = {}
    for i, value in enumerate(coordinates):
        groups.setdefault(find(i), []).append(value)
    groups = list(groups.values())
    lookup = {value: i for i, group in enumerate(groups) for value in group}
    return groups, lookup


class _NativeReferenceCache:
    def __init__(self, accumulator):
        # Keep only canonicalization inputs. Retaining the accumulator here
        # creates observer -> cache -> accumulator -> observer cycles containing
        # the entire completed spectrum.
        self.library_path = accumulator.library_path
        self.generators = accumulator.generators
        self.color_cache = accumulator.color_cache
        self.calls = 0

    def code(self, graph, *, timeout_seconds, max_search_nodes):
        self.calls += 1
        try:
            value, _ = native_canonical_code(
                graph,
                library_path=self.library_path,
                generators=self.generators,
                timeout_seconds=timeout_seconds,
                max_search_nodes=max_search_nodes,
                _its_color_cache=self.color_cache,
                compact=True,
            )
            return PackedCanonicalKey(value), None
        except RuntimeError as error:
            return None, str(error)

    def statistics(self):
        return {"native_reference_calls": self.calls}


class NativeCompactStructureAccumulator(ExactStructureSpectrumAccumulator):
    """Keep exact sparse keys in memory and preserve public class identifiers."""

    @staticmethod
    def _reported_counts(counts):
        return tuple(
            sorted(
                (compact_code_identifier(key), int(count))
                for key, count in counts.items()
            )
        )


class OrbitAccumulator:
    """Weights are |Aut(A)| / |Aut(A) intersect transported Aut(B)|.

    Each ITS canonical key identifies one complete two-sided mapping orbit.
    Coordinate counts average its indicator over reactant coordinate orbits.
    All divisions must be integral; a failed group or canonical proof aborts.
    """

    def __init__(
        self,
        observer,
        generators,
        reactant_order,
        product_order,
        *,
        library_path,
        record_only=False,
    ):
        self.record_only = record_only
        self.observer = observer
        old = observer.structure
        if not isinstance(old, NativeCompactStructureAccumulator):
            observer.structure = NativeCompactStructureAccumulator(
                old.reactant,
                old.product,
                old.elements,
                old.properties,
                enabled=old.enabled,
                template_radius=old.template_radius,
                tolerance=old.tolerance,
                timeout_seconds=old.timeout_seconds,
                max_search_nodes=old.max_search_nodes,
            )
        self.atom_frequencies = Counter()
        self.bond_frequencies = Counter()
        self.generators = generators
        self.reactant_order = reactant_order
        self.product_order = product_order
        self.library_path = library_path
        self.color_cache = {}
        self.its_canonicalizer = NativeITSCanonicalizer(
            observer, generators, library_path
        )
        observer.structure._code_cache = _NativeReferenceCache(self)
        self.seen = {}
        n = len(observer.reactant)
        self.atom_orbits, self.atom_lookup = coordinate_orbits(n, generators)
        self.bond_orbits, self.bond_lookup = coordinate_orbits(
            n, generators, pairs=True
        )
        self.candidates = 0
        self.duplicates = 0
        self.pattern_hits = 0

    def canonical(self, graph):
        return native_canonical_code(
            graph,
            library_path=self.library_path,
            generators=self.generators,
            timeout_seconds=self.observer.structure.timeout_seconds,
            max_search_nodes=self.observer.structure.max_search_nodes,
            compact=True,
            _its_color_cache=self.color_cache,
        )

    def observe(self, mapping, cost):
        self.candidates += 1
        observer = self.observer
        transported = observer.product[np.ix_(mapping, mapping)]
        paired = self.its_canonicalizer.paired_matrix(mapping)
        if self.its_canonicalizer.known_pattern(mapping, paired):
            self.duplicates += 1
            self.pattern_hits += 1
            return
        key, stabilizer = self.its_canonicalizer.canonical(
            mapping, transported, paired=paired, packed=True
        )
        if not isinstance(key, PackedCanonicalKey):
            key = PackedCanonicalKey(key)
        if key in self.seen:
            self.duplicates += 1
            self.its_canonicalizer.remember_pattern()
            return
        self.commit(key, stabilizer, mapping, transported=transported, paired=paired)
        if not self.record_only:
            self.its_canonicalizer.remember_pattern()

    def commit(
        self,
        key,
        stabilizer,
        mapping,
        template_key=None,
        *,
        transported=None,
        paired=None,
    ):
        if key in self.seen:
            return
        observer = self.observer
        if transported is None:
            transported = observer.product[np.ix_(mapping, mapping)]
        if (
            not stabilizer
            or self.reactant_order % stabilizer
            or self.product_order % stabilizer
        ):
            raise RuntimeError("unproved double-orbit multiplicity")
        weight = self.reactant_order // stabilizer
        if paired is None:
            paired = self.its_canonicalizer.paired_matrix(mapping)
        changes = self.its_canonicalizer.changes(mapping, paired)
        if changes is None:
            changed_atoms, changed_bonds = _changed_atoms_and_bonds(
                observer.reactant, transported, observer.properties, mapping,
                observer.tolerance,
            )
            context = _template_context(
                observer.reactant, transported, changed_atoms,
                observer.structure.template_radius,
            )
            atom_changes = [
                i for i in range(len(mapping))
                if any(left[i] != right[mapping[i]]
                       for left, right in observer.properties.values())
            ]
        else:
            atom_changes, changed_bonds, context = changes
        if template_key is None and observer.structure.enabled:
            template_key, _ = self.its_canonicalizer.canonical(
                mapping,
                transported,
                context,
                paired=paired,
                require_group=False,
                packed=True,
            )
            if not isinstance(template_key, PackedCanonicalKey):
                template_key = PackedCanonicalKey(template_key)
        atom_totals = Counter(self.atom_lookup[i] for i in atom_changes)
        bond_totals = Counter(self.bond_lookup[pair] for pair in changed_bonds)
        atom_updates, bond_updates = {}, {}
        for totals, orbits, updates in (
            (atom_totals, self.atom_orbits, atom_updates),
            (bond_totals, self.bond_orbits, bond_updates),
        ):
            for orbit, count in totals.items():
                increment, remainder = divmod(weight * count, len(orbits[orbit]))
                if remainder:
                    raise RuntimeError("nonintegral orbit frequency")
                updates[orbit] = increment
        from ..analysis import _mapping_sha256

        record = (
            weight,
            stabilizer,
            _mapping_sha256(mapping),
            template_key,
            tuple(sorted(atom_updates.items())),
            tuple(sorted(bond_updates.items())),
        )
        if self.record_only:
            self.seen[key] = record
        else:
            self.merge_record(key, record)

    def merge_record(self, key, record):
        """Merge a proved class using precomputed coordinate-orbit totals."""
        if key in self.seen:
            return
        weight, stabilizer, mapping_digest, template_key, atoms, bonds = record
        if (
            not stabilizer
            or self.reactant_order % stabilizer
            or self.product_order % stabilizer
            or weight != self.reactant_order // stabilizer
        ):
            raise RuntimeError("unproved double-orbit merge weight")
        self.seen[key] = (weight, stabilizer)
        observer = self.observer
        observer.count += weight
        self.atom_frequencies.update(dict(atoms))
        self.bond_frequencies.update(dict(bonds))
        payload = str(weight).encode() + b":" + mapping_digest
        observer.stream_digest.update(len(payload).to_bytes(8, "little"))
        observer.stream_digest.update(payload)
        if observer.structure.enabled:
            observer.structure.its_counts[key] += weight
            observer.structure.template_counts[template_key] += weight

    def finish(self):
        """Expand coordinate-orbit totals once, after exact class deduplication."""
        self.observer.atom_counts.clear()
        self.observer.bond_counts.clear()
        for frequencies, orbits, counts in (
            (self.atom_frequencies, self.atom_orbits, self.observer.atom_counts),
            (self.bond_frequencies, self.bond_orbits, self.observer.bond_counts),
        ):
            for orbit, value in frequencies.items():
                for coordinate in orbits[orbit]:
                    counts[coordinate] = value

    def contains_reference(self, mapping):
        observer = self.observer
        transported = observer.product[np.ix_(mapping, mapping)]
        graph = _attributed_its_graph(
            observer.reactant,
            transported,
            observer.structure.elements,
            observer.properties,
            mapping,
        )
        key, _ = self.canonical(graph)
        return PackedCanonicalKey(key) in self.seen
