"""Retained execution conditions for composites on arbitrary hosts.

An endpoint span cannot remember transient edge absence or intermediate
resource tests. This condition transports the supplied sequential witness to
the requested host and requires both applications to be admitted there.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SequentialCondition:
    first: object
    second: object
    first_match: tuple
    first_right_to_intermediate: tuple
    second_match: tuple

    def validate(self, host, outer_match):
        from .dpo import DPOError, DPOIssue, DPOIssueCode, apply_dpo

        try:
            first_match = {node: outer_match[image] for node, image in self.first_match}
            first_application = apply_dpo(self.first, host, first_match)
            first_certificate = first_application.certificate
            transport = {
                node: image
                for node, image in outer_match.items()
                if image not in first_certificate.deleted_nodes
            }
            actual_right = dict(first_certificate.right_to_result)
            for right_node, intermediate_node in self.first_right_to_intermediate:
                transport[intermediate_node] = actual_right[right_node]
            second_match = {node: transport[image] for node, image in self.second_match}
            apply_dpo(self.second, first_application.result, second_match)
        except (DPOError, KeyError) as error:
            raise DPOError(
                DPOIssue(
                    DPOIssueCode.APPLICATION_CONDITION,
                    "The composite's retained sequential application condition failed.",
                    {"reason": str(error)},
                )
            ) from error
