"""C ABI adapter. Matching, completion and bond/resource rewriting run in C++."""

from __future__ import annotations

import ctypes as C
import hashlib
import json
from functools import partial
from pathlib import Path

import networkx as nx

from synkit.Graph.Hyrogen._misc import has_XH
from synkit.Graph.Matcher.subgraph_matcher import resolve_template_match_attrs
from synkit.Graph.Morphism.constraints import EndpointSide, WildcardConstraint, WildcardRole
from ..core.completion import WildcardCompletionError
from ..core.product import _ensure_host_atom_maps
from ..matching.policy import contextual_electron_pattern_graph
from ..matching.wildcard import WildcardSearchLimitError, prepare_wildcard_pattern

Int64 = C.c_int64


def structure(name, fields):
    return type(name, (C.Structure,), {"_fields_": fields})


State = structure("State", [(k, Int64) for k in ("element", "charge", "hcount", "radical", "lp", "ve", "flags")])
Bond = structure("Bond", [(k, Int64) for k in ("order2", "sigma2", "pi2")])
Domain = structure("Domain", [("offset", Int64), ("count", Int64)])
Atom = structure("Atom", [("state", State), ("atom_map", Int64)])
Edge = structure("Edge", [("u", Int64), ("v", Int64), ("state", Bond)])
RuleAtom = structure(
    "RuleAtom",
    [("left", State), ("right", State)]
    + [
        (k, Int64)
        for k in (
            "atom_map",
            "baseline_lp",
            "match_mask",
            "role",
            "owner",
            "capacity",
            "budget",
            "side",
            "identity",
            "materialization",
            "virtual_kind",
            "stereo_slot",
            "fixed_charge",
        )
    ]
    + [(k, Domain) for k in ("elements", "charges", "radicals", "bond_orders")],
)
RuleEdge = structure("RuleEdge", [("u", Int64), ("v", Int64), ("left", Bond), ("right", Bond), ("min_pi2", Int64)])
Request = structure(
    "Request",
    [
        ("atoms", C.POINTER(Atom)),
        ("n_atoms", Int64),
        ("edges", C.POINTER(Edge)),
        ("n_edges", Int64),
        ("rule_atoms", C.POINTER(RuleAtom)),
        ("n_rule_atoms", Int64),
        ("rule_edges", C.POINTER(RuleEdge)),
        ("n_rule_edges", Int64),
        ("domains", C.POINTER(Int64)),
        ("n_domains", Int64),
        ("partial", Int64),
        ("candidate_limit", Int64),
        ("radical_policy", Int64),
    ],
)
ProductAtom = structure(
    "ProductAtom",
    [("left", State), ("right", State)] + [(k, Int64) for k in ("atom_map", "origin", "introduced", "owner", "port")],
)
ProductEdge = structure("ProductEdge", [("u", Int64), ("v", Int64), ("left", Bond), ("right", Bond)])
Stats = structure("Stats", [(k, Int64) for k in ("candidates", "mappings", "applications", "elapsed_ns")])
ROLES = [
    WildcardRole.QUERY_ATOM,
    WildcardRole.ATTACHMENT_PORT,
    WildcardRole.RADICAL_COMPLETION,
    WildcardRole.HYDROGEN_COMPLETION,
    WildcardRole.STEREO_LIGAND_PORT,
]


class NativeReactorUnsupportedError(ValueError):
    """Input lies outside the compiled profile; no Python fallback is used."""


def load(library_path):
    if library_path is None:
        from synkit._native import bundled_library, verify_wheel_library

        path = bundled_library(Path(__file__).parent, "synkit_partial")
        verify_wheel_library(path)
    else:
        path = Path(library_path).expanduser().resolve()
        manifest = json.loads(path.with_suffix(".json").read_text())
        if manifest.get("library_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            raise RuntimeError("Native reactor library integrity check failed.")
    lib = C.CDLL(str(path))
    if lib.skp_abi() != 1:
        raise RuntimeError("Unsupported native reactor ABI.")
    signatures = {
        "skp_execute": ([C.POINTER(Request)], C.c_void_p),
        "skp_status": ([C.c_void_p], C.c_int),
        "skp_error": ([C.c_void_p], C.c_char_p),
        "skp_stats": ([C.c_void_p], C.POINTER(Stats)),
        "skp_application_count": ([C.c_void_p], Int64),
        "skp_mapping": ([C.c_void_p, Int64], C.POINTER(Int64)),
        "skp_atoms": ([C.c_void_p, Int64, C.POINTER(Int64)], C.POINTER(ProductAtom)),
        "skp_edges": ([C.c_void_p, Int64, C.POINTER(Int64)], C.POINTER(ProductEdge)),
        "skp_free": ([C.c_void_p], None),
    }
    if hasattr(lib, "skp_execute_observed"):
        signatures["skp_execute_observed"] = ([C.POINTER(Request)], C.c_void_p)
        signatures["skp_rejected_completion_count"] = ([C.c_void_p], Int64)
    if hasattr(lib, "skp_execute_observed_first"):
        signatures["skp_execute_observed_first"] = ([C.POINTER(Request)], C.c_void_p)
        signatures["skp_search_complete"] = ([C.c_void_p], C.c_int)
    for name, (arguments, result) in signatures.items():
        method = getattr(lib, name)
        method.argtypes, method.restype = arguments, result
    return lib, path


def integer(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise NativeReactorUnsupportedError("Native labels require exact integer resource counts.")
    value = int(value)
    if not -(2**63) <= value < 2**63:
        raise NativeReactorUnsupportedError("A native label exceeds the signed 64-bit representation.")
    return value


def side(value, index):
    return value[index] if isinstance(value, tuple) and len(value) == 2 else value


def _prepare_native_graphs(engine):
    rule = engine.rule
    host = engine._matching_host_graph().copy()
    rc = rule.rc.raw
    observation = rc.graph.get("endpoint_policy") == "observed"
    if observation:
        from synkit.IO.chem_converter import _local_charge_model_consistent

        for node in host:
            if not _local_charge_model_consistent(host, node):
                host.nodes[node]["charge_model_consistent"] = False
    if not observation and any(not side(a.get("present", True), s) for _, a in rc.nodes(data=True) for s in (0, 1)):
        raise NativeReactorUnsupportedError("Native rules must preserve the atom carrier.")
    if engine.explicit_h or engine.preserve_mapped_hydrogens or has_XH(rule.left.raw):
        raise NativeReactorUnsupportedError("Native profile requires implicit heavy-atom hydrogens.")
    if (
        rule.stereo_guards
        or rule.stereo_effects
        or rule.stereo_outcomes
        or rule.stereo_couplings
        or host.graph.get("stereo_descriptors")
    ):
        raise NativeReactorUnsupportedError("Stereo-bearing applications are not yet compiled.")
    if (
        host.is_directed()
        or host.is_multigraph()
        or nx.number_of_selfloops(host)
        or any(a.get("element") == "*" for _, a in host.nodes(data=True))
    ):
        raise NativeReactorUnsupportedError("Native matching requires a concrete simple undirected host.")
    if any(attrs.get("isotope", 0) for _, attrs in host.nodes(data=True)):
        raise NativeReactorUnsupportedError("Isotope-labelled applications are not yet compiled.")
    _ensure_host_atom_maps(host)
    return rule, host, rc, observation


def _element_codes(host, rc):
    elements = {"*"}
    for graph in (host, rc):
        for _, attrs in graph.nodes(data=True):
            for index in (0, 1):
                elements.add(side(attrs["element"], index))
            elements.update(attrs.get("elements") or ())
    symbols = ["*"] + sorted(elements - {"*"})
    codes = {value: i for i, value in enumerate(symbols)}

    return symbols, codes


def _atom_state(attrs, index=0, *, codes):
    for required in ("element", "charge", "hcount", "radical", "lone_pairs", "valence_electrons"):
        if required not in attrs:
            raise NativeReactorUnsupportedError("Native atoms require a complete Lewis state.")
    present = bool(side(attrs.get("present", True), index))
    aromatic = bool(side(attrs.get("aromatic", False), index))
    return State(
        codes[side(attrs["element"], index)],
        integer(side(attrs["charge"], index)),
        integer(side(attrs["hcount"], index)),
        integer(side(attrs["radical"], index)),
        integer(side(attrs["lone_pairs"], index)),
        integer(side(attrs["valence_electrons"], index)),
        int(present) | (2 if aromatic else 0) | (4 if present else 0),
    )


def _bond_state(attrs, index=0):
    order = side(attrs.get("order", 0), index)
    sigma = side(attrs.get("sigma_order", 1 if order else 0), index)
    pi = side(attrs.get("pi_order", max(0, order - sigma)), index)
    return Bond(integer(2 * order), integer(2 * sigma), integer(2 * pi))


def compile_native_request(engine):
    """Compile exact ABI records, retaining ownership for pointer-backed arrays.

    This is shared by direct reactor execution and native CRN controllers.
    It performs no matching, rewriting, library loading or reaction execution.
    """
    rule, host, rc, observation = _prepare_native_graphs(engine)
    pattern = contextual_electron_pattern_graph(
        engine._relative_pi_pattern_graph(rule.left.raw), reaction_center=rc, template_format="tuple"
    )
    pattern = prepare_wildcard_pattern(pattern)
    host_nodes, rule_nodes = tuple(host), tuple(rc)
    hi = {node: i for i, node in enumerate(host_nodes)}
    ri = {node: i for i, node in enumerate(rule_nodes)}
    symbols, codes = _element_codes(host, rc)

    state = partial(_atom_state, codes=codes)
    bond = _bond_state

    domain_values = []

    def domain(values, transform=integer):
        if values is None:
            return Domain(0, -1)
        offset = len(domain_values)
        domain_values.extend(transform(value) for value in sorted(values))
        return Domain(offset, len(domain_values) - offset)

    attributes, _ = resolve_template_match_attrs(pattern)
    rule_atoms = []
    by_map = {side(attrs.get("atom_map", node), 0): node for node, attrs in rc.nodes(data=True)}
    for node in rule_nodes:
        attrs = rc.nodes[node]
        if node not in pattern and not observation:
            raise NativeReactorUnsupportedError("Native rules must preserve the complete atom carrier.")
        query = pattern.nodes[node] if node in pattern else {}
        constraint = query.get("_wildcard_constraint")
        if constraint is not None and constraint.role is WildcardRole.STEREO_LIGAND_PORT:
            raise NativeReactorUnsupportedError("Typed stereo-port assignments are not yet compiled.")
        owner = constraint.owner if constraint else None
        owner_node = owner if owner in ri else by_map.get(owner)
        mask = 0
        for name, bit in (
            ("element", 1),
            ("charge", 2),
            ("hcount", 4),
            ("radical", 8),
            ("lone_pairs", 16),
            ("aromatic", 32),
        ):
            if name in attributes and query.get("_query_attribute_policies", {}).get(name) != "unknown":
                mask |= bit
        if engine.radical_policy == "ignore":
            mask &= ~8
        consistency = attrs.get("charge_model_consistent")
        fixed = isinstance(consistency, tuple) and (consistency[1] is False if observation else not all(consistency))
        if (
            observation
            and isinstance(consistency, tuple)
            and any(side(attrs.get("present", True), s) and consistency[s] is False for s in (0, 1))
        ):
            fixed = int(fixed) | 2
        left, right = state(attrs, 0), state(attrs, 1)
        rule_atoms.append(
            RuleAtom(
                left,
                right,
                integer(side(attrs.get("atom_map", 0), 0)),
                integer(attrs.get("_relative_resource_baselines", {}).get("lone_pairs", 0)),
                mask,
                ROLES.index(constraint.role) if constraint else -1,
                ri[owner_node] if owner_node is not None else -1,
                constraint.capacity if constraint else 1,
                constraint.resource_budget if constraint and constraint.resource_budget is not None else -1,
                (
                    {EndpointSide.ANY: 0, EndpointSide.REACTANT: 1, EndpointSide.PRODUCT: 2}[constraint.side]
                    if constraint
                    else 0
                ),
                constraint.mapped_identity if constraint and constraint.mapped_identity is not None else -1,
                {None: 0, "concrete": 1, "virtual": 2}.get(constraint.materialization, 2) if constraint else 0,
                {None: 0, "H": 1, "LP": 2}[constraint.virtual_kind] if constraint else 0,
                constraint.stereo_slot if constraint and constraint.stereo_slot is not None else -1,
                int(fixed),
                domain(constraint.elements if constraint else None, lambda value: codes[value]),
                domain(constraint.charges if constraint else None),
                domain(constraint.radicals if constraint else None),
                domain(constraint.bond_orders if constraint else None, lambda value: integer(2 * value)),
            )
        )
    rule_edges = []
    for left, right, attrs in rc.edges(data=True):
        match = pattern.get_edge_data(left, right, {})
        minimum = match.get("_minimum_pi_order")
        a, b = bond(attrs, 0), bond(attrs, 1)
        rule_edges.append(RuleEdge(ri[left], ri[right], a, b, integer(2 * minimum) if minimum is not None else -1))
    arrays = (
        (Atom * len(host_nodes))(*(Atom(state(host.nodes[n]), integer(host.nodes[n]["atom_map"])) for n in host_nodes)),
        (Edge * host.number_of_edges())(*(Edge(hi[u], hi[v], bond(a)) for u, v, a in host.edges(data=True))),
        (RuleAtom * len(rule_atoms))(*rule_atoms),
        (RuleEdge * len(rule_edges))(*rule_edges),
        (Int64 * len(domain_values))(*domain_values),
    )
    request = Request(
        arrays[0],
        len(arrays[0]),
        arrays[1],
        len(arrays[1]),
        arrays[2],
        len(arrays[2]),
        arrays[3],
        len(arrays[3]),
        arrays[4],
        len(arrays[4]),
        int(engine.partial),
        engine.embed_threshold if engine.embed_threshold is not None else -1,
        {"strict": 0, "lower_bound": 1, "ignore": 2}[engine.radical_policy],
    )
    return {
        "request": request,
        "arrays": arrays,
        "host": host,
        "rc": rc,
        "pattern": pattern,
        "host_nodes": host_nodes,
        "rule_nodes": rule_nodes,
        "symbols": symbols,
        "hi": hi,
        "observation": observation,
    }


def serialize_native_request(engine):
    """Return JSON-safe ABI records using the direct reactor's exact compiler.

    Element integers index the accompanying symbols table. A native controller
    must remap that table consistently when composing hosts with rule records.
    """
    compiled = compile_native_request(engine)
    arrays = compiled["arrays"]
    request = compiled["request"]

    def record(value):
        if isinstance(value, C.Structure):
            return {name: record(getattr(value, name)) for name, _ in value._fields_}
        return int(value)

    return {
        "symbols": compiled["symbols"],
        "atoms": [record(value) for value in arrays[0]],
        "edges": [record(value) for value in arrays[1]],
        "rule_atoms": [record(value) for value in arrays[2]],
        "rule_edges": [record(value) for value in arrays[3]],
        "domains": [int(value) for value in arrays[4]],
        "partial": bool(request.partial),
        "candidate_limit": int(request.candidate_limit),
        "radical_policy": int(request.radical_policy),
    }


def _product_node_attributes(atom, i, names, compiled):
    host, rc, pattern = compiled["host"], compiled["rc"], compiled["pattern"]
    host_nodes, rule_nodes = compiled["host_nodes"], compiled["rule_nodes"]
    symbols = compiled["symbols"]
    attrs = dict(host.nodes[host_nodes[i]]) if i < len(host_nodes) else {}
    a, b = atom.left, atom.right
    attrs.update(
        {
            "element": (symbols[a.element], symbols[b.element]),
            "charge": (a.charge, b.charge),
            "hcount": (a.hcount, b.hcount),
            "radical": (a.radical, b.radical),
            "lone_pairs": (a.lp, b.lp),
            "aromatic": (bool(a.flags & 2), bool(b.flags & 2)),
            "atom_map": (atom.atom_map, atom.atom_map),
            "present": (bool(a.flags & 1), bool(b.flags & 1)),
        }
    )
    attrs["typesGH"] = (
        (symbols[a.element], bool(a.flags & 2), a.hcount, a.charge, []),
        (symbols[b.element], bool(b.flags & 2), b.hcount, b.charge, []),
    )
    if (a.flags | b.flags) & 4:
        attrs["valence_electrons"] = (a.ve, b.ve)
    if atom.introduced:
        attrs["_rule_completion_atom"] = not bool(atom.port)
    if atom.port:
        constraint = WildcardConstraint("attachment_port", owner=names[atom.owner], bond_orders={1}, resource_budget=1)
    elif a.flags & 1 and a.element == 0 and atom.origin >= 0:
        constraint = pattern.nodes[rule_nodes[atom.origin]]["_wildcard_constraint"]
        if constraint.owner is not None:
            constraint = constraint.relabel_owner({constraint.owner: names[atom.owner]})
    else:
        constraint = None
    if constraint is not None:
        attrs["wildcard_role"] = constraint.role.value
        attrs["_wildcard_constraint"] = constraint
        for name in (
            "elements",
            "charges",
            "radicals",
            "bond_orders",
            "side",
            "owner",
            "capacity",
            "resource_budget",
            "mapped_identity",
            "materialization",
        ):
            value = getattr(constraint, name)
            if value is not None:
                attrs[name] = value
    if atom.origin >= 0:
        source_attrs = rc.nodes[rule_nodes[atom.origin]]
        target_charge = b.charge if source_attrs["element"] == ("*", "*") else side(source_attrs["charge"], 1)
        attrs["template_charge"] = (a.charge, target_charge)
        if "charge_model_consistent" in source_attrs:
            attrs["charge_model_consistent"] = source_attrs["charge_model_consistent"]
    return attrs


def _decode_application(lib, result, application, compiled):
    host_nodes, rule_nodes = compiled["host_nodes"], compiled["rule_nodes"]
    hi = compiled["hi"]
    observation = compiled["observation"]
    raw = lib.skp_mapping(result, application)
    mapping = {node: host_nodes[raw[i]] for i, node in enumerate(rule_nodes) if raw[i] >= 0}
    count = Int64()
    native_atoms = lib.skp_atoms(result, application, C.byref(count))
    atoms = [ProductAtom.from_buffer_copy(native_atoms[i]) for i in range(count.value)]
    native_edges = lib.skp_edges(result, application, C.byref(count))
    edges = [ProductEdge.from_buffer_copy(native_edges[i]) for i in range(count.value)]
    names = list(host_nodes)
    next_id = max((n for n in host_nodes if isinstance(n, int)), default=0) + 1
    while len(names) < len(atoms):
        while next_id in hi:
            next_id += 1
        names.append(next_id)
        next_id += 1
    graph = nx.Graph()
    for i, atom in enumerate(atoms):
        attrs = _product_node_attributes(atom, i, names, compiled)
        graph.add_node(names[i], **attrs)
    for edge in edges:
        a, b = edge.left, edge.right
        graph.add_edge(
            names[edge.u],
            names[edge.v],
            order=(a.order2 / 2, b.order2 / 2),
            sigma_order=(a.sigma2 / 2, b.sigma2 / 2),
            pi_order=(a.pi2 / 2, b.pi2 / 2),
            kekule_order=((a.sigma2 + a.pi2) / 2, (b.sigma2 + b.pi2) / 2),
            standard_order=(a.order2 - b.order2) / 2,
        )
    for node, attrs in graph.nodes(data=True):
        if "valence_electrons" not in attrs or not attrs["present"][1]:
            continue
        bond_sum = sum(e["sigma_order"][1] + e["pi_order"][1] for _, _, e in graph.edges(node, data=True))
        charge = (
            attrs["valence_electrons"][1]
            - 2 * attrs["lone_pairs"][1]
            - attrs["radical"][1]
            - attrs["hcount"][1]
            - bond_sum
        )
        attrs["recomputed_charge"] = (None, charge)
        expected = attrs.get("template_charge", attrs["charge"])[1]
        attrs["charge_mismatch"] = (None, charge != expected)
    introduced = [i for i, atom in enumerate(atoms) if atom.introduced and not atom.port]
    if introduced:
        graph.graph["partial_completion"] = {
            "schema": "synkit.partial-completion/1",
            "introduced_atoms": tuple(names[i] for i in introduced),
            "introduced_mapping": tuple((rule_nodes[atoms[i].origin], names[i]) for i in introduced),
            "attachment_ports": tuple(names[i] for i, atom in enumerate(atoms) if atom.port),
        }
    graph.graph.update(
        electron_aware_rewrite=True,
        wildcard_mode="typed",
        endpoint_policy="observed" if observation else "closed",
        _product_electron_fields_current=True,
        _product_kekule_phase_dirty=False,
    )
    return mapping, graph


def execute(engine):
    """Compile records, invoke one native atomic batch, and decode paired graphs."""
    compiled = compile_native_request(engine)
    request = compiled["request"]
    observation = compiled["observation"]
    # CRN builders retain a verified handle across rule applications.
    preloaded = getattr(engine, "_preloaded_native_library", None)
    lib, library = preloaded if preloaded is not None else load(engine.library_path)
    if observation and not hasattr(lib, "skp_execute_observed"):
        raise NativeReactorUnsupportedError("This native build does not implement observation-aware execution.")
    if engine.first_application:
        if not observation or not hasattr(lib, "skp_execute_observed_first"):
            raise NativeReactorUnsupportedError("First-application execution needs the native observation policy.")
        method = lib.skp_execute_observed_first
    else:
        method = lib.skp_execute_observed if observation else lib.skp_execute
    result = method(C.byref(request))
    try:
        if lib.skp_status(result):
            message = lib.skp_error(result).decode()
            exception = {1: NativeReactorUnsupportedError, 2: WildcardSearchLimitError, 3: WildcardCompletionError}[
                lib.skp_status(result)
            ]
            raise exception(message)
        statistics = lib.skp_stats(result).contents
        statistics = {name: getattr(statistics, name) for name, _ in Stats._fields_}
        statistics["search_complete"] = (
            bool(lib.skp_search_complete(result)) if hasattr(lib, "skp_search_complete") else True
        )
        if hasattr(lib, "skp_rejected_completion_count"):
            statistics["rejected_completions"] = lib.skp_rejected_completion_count(result)
        mappings, graphs = [], []
        for application in range(lib.skp_application_count(result)):
            mapping, graph = _decode_application(lib, result, application, compiled)
            mappings.append(mapping)
            graphs.append(graph)
        return {"mappings": mappings, "graphs": graphs, "statistics": {**statistics, "library": str(library), "abi": 1}}
    finally:
        lib.skp_free(result)
