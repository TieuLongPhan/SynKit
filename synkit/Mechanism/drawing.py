"""Headless static overview figures for versioned mechanism records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .model import MechanismRecord, VerificationCertificate


def draw_mechanism_record(
    record: MechanismRecord,
    *,
    certificate: VerificationCertificate | None = None,
    path: str | Path | None = None,
) -> Any:
    """Draw grouped electron events and stereo effects as an SVG/PDF-ready figure."""
    rows = max(1, len(record.steps))
    figure, axes = plt.subplots(
        rows, 1, figsize=(10, max(2.4, rows * 2.1)), squeeze=False
    )
    issue_steps = (
        {issue.step_id for issue in certificate.issues} if certificate else set()
    )
    for index, step in enumerate(record.steps or (None,)):
        axis = axes[index][0]
        axis.set_axis_off()
        if step is None:
            axis.text(0.5, 0.5, "No declared steps", ha="center", va="center")
            continue
        failing = step.step_id in issue_steps
        axis.set_title(
            step.step_id, loc="left", color="#B42318" if failing else "#172B4D"
        )
        events = [move for group in step.groups for move in group.moves]
        count = max(1, len(events))
        for event_index, move in enumerate(events):
            y = 0.78 - event_index * (0.55 / count)
            source = f"{move.source.kind}({','.join(map(str, move.source.atom_maps))})"
            target = f"{move.target.kind}({','.join(map(str, move.target.atom_maps))})"
            arrow = "fishhook" if move.electron_count == 1 else "curved"
            axis.annotate(
                "",
                xy=(0.69, y),
                xytext=(0.31, y),
                arrowprops={
                    "arrowstyle": "-|>",
                    "lw": 1.2 if arrow == "fishhook" else 2.0,
                },
            )
            axis.text(0.29, y, source, ha="right", va="center", family="monospace")
            axis.text(0.71, y, target, ha="left", va="center", family="monospace")
            axis.text(
                0.5, y + 0.055, f"{arrow} · {move.group_id}", ha="center", fontsize=8
            )
        badges = "  ".join(
            f"[{effect.effect.lower()} {effect.descriptor_target[0]}:{effect.descriptor_target[1]}]"
            for effect in step.stereo_effects
        )
        if badges:
            axis.text(0.5, 0.08, badges, ha="center", color="#7A3E00", fontsize=9)
        if failing:
            axis.add_patch(
                plt.Rectangle(
                    (0.01, 0.01), 0.98, 0.96, fill=False, color="#D92D20", lw=1.5
                )
            )
    figure.suptitle(record.mapped_reaction, fontsize=10)
    figure.tight_layout()
    if path is not None:
        output = Path(path)
        if output.suffix.lower() not in {".svg", ".pdf", ".png"}:
            raise ValueError("Mechanism figures must use .svg, .pdf, or .png.")
        figure.savefig(output, bbox_inches="tight")
    return figure
