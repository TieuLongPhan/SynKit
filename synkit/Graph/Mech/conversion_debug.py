from __future__ import annotations

from typing import Any

from .conversion import (
    build_its_from_rsmi,
    classify_arrow_shape,
    convert_reaction_arrow,
    get_its_bond_order,
    parse_arrow_step,
    remove_non_arrow_atom_maps,
    split_arrow_code,
    validate_arrow_maps,
)


def debug_arrow_bond_orders(
    reaction_smiles: str,
    arrow_code: str,
    expand_aam: bool = True,
    remove_non_arrow_maps: bool = True,
    strict_bond_lookup: bool = True,
) -> None:
    """Print the ITS bond orders used by each arrow step.

    :param reaction_smiles: Reaction SMILES in ``reactants>>products`` format.
    :type reaction_smiles: str
    :param arrow_code: Semicolon-separated arrow code.
    :type arrow_code: str
    :param expand_aam: Whether to expand atom mapping before ITS construction.
    :type expand_aam: bool
    :param remove_non_arrow_maps: Whether to remove atom maps not used by the arrow code.
    :type remove_non_arrow_maps: bool
    :param strict_bond_lookup: Whether missing typed bond lookups should raise.
    :type strict_bond_lookup: bool
    :returns: ``None``.
    :rtype: None
    """
    its, expanded_rsmi, rsmi_for_its, diagnostics = build_its_from_rsmi(
        rsmi=reaction_smiles,
        arrow_code=arrow_code,
        expand_aam=expand_aam,
        remove_non_arrow_maps=remove_non_arrow_maps,
    )

    print("Diagnostics:")
    print(diagnostics)
    print()

    print("RSMI used for ITS:")
    print(rsmi_for_its)
    print()

    print("Expanded RSMI:")
    print(expanded_rsmi)
    print()

    for step in split_arrow_code(arrow_code):
        lhs, rhs = parse_arrow_step(step)

        print(f"Step: {step}")
        print(f"  shape: {classify_arrow_shape(step)}")

        if len(lhs) == 1 and len(rhs) == 1:
            a = lhs[0]
            b = rhs[0]
            print(
                f"  destination bond {a}-{b}: "
                f"{get_its_bond_order(its, a, b, strict=strict_bond_lookup, context=step)}"
            )

        elif len(lhs) == 1 and len(rhs) == 2:
            a = lhs[0]
            b, c = rhs
            print(f"  LP source atom    {a}")
            print(
                f"  destination bond {b}-{c}: "
                f"{get_its_bond_order(its, b, c, strict=strict_bond_lookup, context=step)}"
            )

        elif len(lhs) == 2 and len(rhs) == 1:
            a, b = lhs
            c = rhs[0]
            print(
                f"  source bond      {a}-{b}: "
                f"{get_its_bond_order(its, a, b, strict=strict_bond_lookup, context=step)}"
            )
            print(f"  LP destination atom {c}")

        elif len(lhs) == 2 and len(rhs) == 2:
            a, b = lhs
            c, d = rhs
            print(
                f"  source bond      {a}-{b}: "
                f"{get_its_bond_order(its, a, b, strict=strict_bond_lookup, context=step)}"
            )
            print(
                f"  destination bond {c}-{d}: "
                f"{get_its_bond_order(its, c, d, strict=strict_bond_lookup, context=step)}"
            )

        print()


def debug_record(
    record: dict[str, Any],
    reaction_key: str = "SMIRKS",
    arrow_key: str = "arrow_code",
    orbital_key: str = "orbital pair classification",
) -> dict[str, Any]:
    """Print full debug output for one record.

    :param record: Source record to inspect.
    :type record: dict[str, Any]
    :param reaction_key: Key containing reaction SMILES.
    :type reaction_key: str
    :param arrow_key: Key containing arrow code.
    :type arrow_key: str
    :param orbital_key: Key containing optional orbital classification metadata.
    :type orbital_key: str
    :returns: Converted record.
    :rtype: dict[str, Any]
    """
    from pprint import pprint

    rsmi = record[reaction_key]
    arrow_code = record[arrow_key]
    orbital_class = record.get(orbital_key)

    print("=" * 100)
    print("DEBUG RECORD")
    print("=" * 100)

    print("orbital_class:")
    print(orbital_class)
    print()

    print("arrow_code:")
    print(arrow_code)
    print()

    print("arrow shapes:")
    for step in split_arrow_code(arrow_code):
        print(f"  {step:25s} -> {classify_arrow_shape(step)}")
    print()

    print("raw map diagnostics:")
    diagnostics = validate_arrow_maps(
        rsmi=rsmi,
        arrow_code=arrow_code,
        raise_on_arrow_duplicates=False,
        raise_on_missing_arrow_maps=False,
    )
    pprint(diagnostics, width=160)
    print()

    print("RSMI after removing non-arrow maps:")
    cleaned = remove_non_arrow_atom_maps(rsmi, arrow_code)
    print(cleaned)
    print()

    print("Arrow bond orders:")
    debug_arrow_bond_orders(
        reaction_smiles=rsmi,
        arrow_code=arrow_code,
        expand_aam=True,
        remove_non_arrow_maps=True,
        strict_bond_lookup=True,
    )

    result = convert_reaction_arrow(
        reaction_smiles=rsmi,
        arrow_code=arrow_code,
        orbital_class=orbital_class,
        expand_aam=True,
        remove_non_arrow_maps=True,
        strict_bond_lookup=True,
    )

    print("converted:")
    pprint(result["converted"], width=120)
    print()

    print("typed_converted:")
    pprint(result["typed_converted"], width=120)
    print()

    return result


def check_typed_conversion_quality(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Check whether typed conversions still contain generic B-/B+ labels.

    :param results: Conversion results to inspect.
    :type results: list[dict[str, Any]]
    :returns: Error and untyped-step diagnostics.
    :rtype: dict[str, Any]
    """
    errors = []
    untyped = []

    for result in results:
        if "error" in result:
            errors.append(result)
            continue

        typed = result.get("typed_converted")

        if typed is None:
            untyped.append(
                {
                    "row_index": result.get("row_index"),
                    "reason": "typed_converted is None",
                }
            )
            continue

        for step in typed:
            label = step[0]
            if "B-" in label or "B+" in label:
                untyped.append(
                    {
                        "row_index": result.get("row_index"),
                        "orbital_class": result.get("orbital_class"),
                        "arrow_code": result.get("arrow_code"),
                        "step": step,
                    }
                )

    return {
        "n_results": len(results),
        "n_errors": len(errors),
        "n_untyped_steps": len(untyped),
        "all_fully_typed": len(errors) == 0 and len(untyped) == 0,
        "errors": errors,
        "untyped": untyped,
    }
