"""Independent finite checks for injective natural DPO squares.

The checker uses carrier coverage, exact intersection, coordinate unions and
coordinate agreement. It does not assume an adhesive ambient category.
"""

from __future__ import annotations

from synkit.Graph.Morphism import (
    LLGError,
    derive_electron_labeled_graph,
    is_electron_schema,
)


def _image_edges(graph, mapping):
    return {frozenset(mapping[node] for node in edge) for edge in graph.edge_keys}


def _morphism(source, target, pairs, name, issues):
    from .dpo import DPOIssue, DPOIssueCode

    mapping = dict(pairs)
    if (
        len(mapping) != len(pairs)
        or set(mapping) != set(source.node_ids)
        or len(set(mapping.values())) != len(mapping)
        or not set(mapping.values()) <= set(target.node_ids)
    ):
        issues.append(
            DPOIssue(
                DPOIssueCode.NONCOMMUTATIVE,
                "A certificate map is not a total injective carrier map.",
                {"map": name},
            )
        )
        return None
    for node, image in mapping.items():
        labels = source.node_labels(node, semantic=True)
        target_labels = target.node_labels(image, semantic=True)
        if any(
            key not in target_labels or target_labels[key] != value
            for key, value in labels.items()
        ):
            issues.append(
                DPOIssue(
                    DPOIssueCode.MATCH_LABEL,
                    "A certificate map does not preserve defined node labels.",
                    {"map": name},
                )
            )
    for edge in source.edge_keys:
        image = frozenset(mapping[node] for node in edge)
        if image not in target.edge_keys:
            issues.append(
                DPOIssue(
                    DPOIssueCode.MATCH_INCIDENCE,
                    "A certificate map does not preserve incidence.",
                    {"map": name},
                )
            )
            continue
        labels = source.edge_labels(edge, semantic=True)
        target_labels = target.edge_labels(image, semantic=True)
        if any(
            key not in target_labels or target_labels[key] != value
            for key, value in labels.items()
        ):
            issues.append(
                DPOIssue(
                    DPOIssueCode.MATCH_LABEL,
                    "A certificate map does not preserve defined edge labels.",
                    {"map": name},
                )
            )
    return mapping


def _square(
    interface,
    first,
    context,
    target,
    arm,
    to_context,
    first_target,
    context_target,
    issues,
):
    from .dpo import DPOIssue, DPOIssueCode

    for edge_mode in (False, True):
        if edge_mode:
            first_image = _image_edges(first, first_target)
            context_image = _image_edges(context, context_target)
            interface_image = _image_edges(
                interface, {n: first_target[arm[n]] for n in interface.node_ids}
            )
            target_carriers = set(target.edge_keys)
            first_carriers = first.edge_keys
            context_carriers = context.edge_keys
            interface_carriers = interface.edge_keys
            keys = target.schema.semantic_edge_keys

            def labels(graph, item):
                return graph.edge_labels(item, semantic=True)

            def transport(mapping, item):
                return frozenset(mapping[n] for n in item)

        else:
            first_image = set(first_target.values())
            context_image = set(context_target.values())
            interface_image = {first_target[arm[n]] for n in interface.node_ids}
            target_carriers = set(target.node_ids)
            first_carriers = first.node_ids
            context_carriers = context.node_ids
            interface_carriers = interface.node_ids
            keys = target.schema.semantic_node_keys

            def labels(graph, item):
                return graph.node_labels(item, semantic=True)

            def transport(mapping, item):
                return mapping[item]

        if (
            first_image | context_image != target_carriers
            or first_image & context_image != interface_image
        ):
            issues.append(
                DPOIssue(
                    DPOIssueCode.NONCOMMUTATIVE,
                    "A square fails carrier coverage or exact interface intersection.",
                    {"edges": edge_mode},
                )
            )
            continue
        _label_square(
            interface,
            first,
            context,
            target,
            arm,
            to_context,
            first_target,
            context_target,
            labels,
            transport,
            keys,
            first_carriers,
            context_carriers,
            target_carriers,
            interface_carriers,
            issues,
        )


def _label_square(
    interface,
    first,
    context,
    target,
    arm,
    to_context,
    first_target,
    context_target,
    labels,
    transport,
    keys,
    first_carriers,
    context_carriers,
    target_carriers,
    interface_carriers,
    issues,
):
    from .dpo import DPOIssue, DPOIssueCode

    supplied = {}
    for graph, carriers, mapping in (
        (first, first_carriers, first_target),
        (context, context_carriers, context_target),
    ):
        for item in carriers:
            image = transport(mapping, item)
            union = supplied.setdefault(image, {})
            for key, value in labels(graph, item).items():
                if key in union and union[key] != value:
                    issues.append(
                        DPOIssue(DPOIssueCode.RESULT_LABEL, "Pushout labels conflict.")
                    )
                union[key] = value
    for item in target_carriers:
        if supplied[item] != labels(target, item):
            issues.append(
                DPOIssue(
                    DPOIssueCode.RESULT_LABEL,
                    "Target labels are not the union of the two arms.",
                )
            )
    for item in interface_carriers:
        first_labels = labels(first, transport(arm, item))
        context_labels = labels(context, transport(to_context, item))
        agreement = {
            key: first_labels[key]
            for key in keys
            if key in first_labels
            and key in context_labels
            and first_labels[key] == context_labels[key]
        }
        if labels(interface, item) != agreement:
            issues.append(
                DPOIssue(
                    DPOIssueCode.RESULT_LABEL,
                    "The interface is not the coordinate agreement pullback.",
                )
            )


def validate_certificate(certificate):
    """Check map shape, labels, naturality, resources, policy and edit metadata."""
    from .dpo import DPOError, DPOIssue, DPOIssueCode, RuleEmbedding

    c = certificate
    issues = []
    schema = c.rule.left.schema
    if any(
        graph.schema != schema
        for graph in (c.rule.interface, c.rule.right, c.host, c.context, c.result)
    ):
        return [
            DPOIssue(
                DPOIssueCode.SCHEMA_MISMATCH, "All square objects must use one schema."
            )
        ]
    if c.host.partial or c.result.partial or not c.context.partial:
        return [
            DPOIssue(
                DPOIssueCode.INTERFACE_PARTIAL,
                "Host and result are total; context is partial.",
            )
        ]
    try:
        RuleEmbedding(c.rule.interface, c.rule.left, c.rule.left_arm.f)
        RuleEmbedding(c.rule.interface, c.rule.right, c.rule.right_arm.f)
        c.rule.__post_init__()
    except DPOError as error:
        return list(error.issues)
    maps = [
        _morphism(source, target, pairs, name, issues)
        for source, target, pairs, name in (
            (c.rule.left, c.host, c.match, "match"),
            (
                c.rule.interface,
                c.context,
                c.interface_to_context,
                "interface_to_context",
            ),
            (c.context, c.host, c.context_to_host, "context_to_host"),
            (c.context, c.result, c.context_to_result, "context_to_result"),
            (c.rule.right, c.result, c.right_to_result, "right_to_result"),
        )
    ]
    if issues:
        return issues
    match, to_context, context_host, context_result, right_result = maps
    for arm, first_target, context_target in (
        (c.rule.left_arm.mapping, match, context_host),
        (c.rule.right_arm.mapping, right_result, context_result),
    ):
        if any(
            first_target[arm[n]] != context_target[to_context[n]]
            for n in c.rule.interface.node_ids
        ):
            return [
                DPOIssue(
                    DPOIssueCode.NONCOMMUTATIVE,
                    "A certificate square does not commute.",
                )
            ]
    _square(
        c.rule.interface,
        c.rule.left,
        c.context,
        c.host,
        c.rule.left_arm.mapping,
        to_context,
        match,
        context_host,
        issues,
    )
    _square(
        c.rule.interface,
        c.rule.right,
        c.context,
        c.result,
        c.rule.right_arm.mapping,
        to_context,
        right_result,
        context_result,
        issues,
    )
    _validate_edits_and_condition(c, match, context_result, issues)
    if c.resource_delta != c.rule.resource_delta:
        issues.append(
            DPOIssue(
                DPOIssueCode.OPEN_TOKEN_MISMATCH,
                "Recorded resource delta is incorrect.",
            )
        )
    if is_electron_schema(schema):
        for graph in (c.host, c.result):
            try:
                derive_electron_labeled_graph(graph.to_networkx())
            except LLGError:
                issues.append(
                    DPOIssue(
                        DPOIssueCode.RESULT_LABEL,
                        "A total electron host has incoherent derived labels.",
                    )
                )
    return issues


def _validate_edits_and_condition(c, match, context_result, issues):
    from .dpo import DPOError, DPOIssue, DPOIssueCode, _deletion_plan

    try:
        deletion = _deletion_plan(c.rule, c.host, match)
        if c.rule.condition is not None:
            c.rule.condition.validate(c.host, match)
    except DPOError as error:
        issues.extend(error.issues)
    else:
        expected_added_nodes = set(c.result.node_ids) - set(context_result.values())
        expected_added_edges = set(c.result.edge_keys) - _image_edges(
            c.context, context_result
        )
        metadata = (
            (c.deleted_nodes, deletion.nodes),
            (c.deleted_edges, deletion.edges),
            (c.added_nodes, expected_added_nodes),
            (c.added_edges, expected_added_edges),
        )
        if any(
            len(items) != len(set(items)) or set(items) != set(expected)
            for items, expected in metadata
        ):
            issues.append(
                DPOIssue(
                    DPOIssueCode.NONCOMMUTATIVE,
                    "Edit provenance does not match the squares.",
                )
            )
