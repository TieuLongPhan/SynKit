# synkit/CRN/tools/enumerator.py
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from synkit.CRN import ReactionNetwork, ReactionPathwayExplorer

from .exceptions import NoPathwaysError
from .helpers import (
    dedupe_pathways_by_canonical,
    pretty_print_pathway,
    replay_pathway_and_collect_inferred,
)


class MotifEnumerator:
    """
    Class responsible for enumerating pathways for motifs.

    Usage:
        enumerator = MotifEnumerator(max_depth=50)
        summary = enumerator.enumerate_motif("motif_1", motif_1, motif_1_config)
    """

    def __init__(self, *, max_depth: int = 50, max_pathways: int = 2000) -> None:
        """
        :param max_depth: Depth limit for pathway enumeration.
        :param max_pathways: Maximum pathways to collect per motif.
        """
        self.max_depth = int(max_depth)
        self.max_pathways = int(max_pathways)

    def enumerate_motif(
        self,
        name: str,
        reactions: List[str],
        config: Optional[Dict[str, Any]] = None,
        *,
        infer_missing: bool = True,
        show_n: int = 3,
    ) -> Dict[str, Any]:
        """
        Enumerate forward pathways for a single motif.

        :param name: Human name for motif.
        :param reactions: List of reaction strings.
        :param config: Optional motif config dict (may contain "sources" and "sinks").
        :param infer_missing: If True, allow guarded inference of missing co-reactants.
        :param show_n: Number of example pathways to print.
        :raises NoPathwaysError: if no unique pathways discovered for this motif.
        :returns: Summary dictionary with keys: name, n_raw, n_unique, examples, net, paths.
        """
        print(f"\n=== Enumerating {name} (len={len(reactions)}) ===")

        net = ReactionNetwork.from_raw_list(
            reactions, standardizer=None, remove_aam=True
        )

        # build start: prefer config["sources"] if present, otherwise detect Source.* lines
        start = Counter()
        if config and isinstance(config.get("sources"), dict) and config["sources"]:
            for token in config["sources"].keys():
                start[token] += 1
        else:
            for r in reactions:
                if isinstance(r, str) and r.startswith("Source."):
                    start[r.split(">>", 1)[0]] += 1

        # sinks from config or default => Removed
        sink_set = {"Removed"}
        if config and isinstance(config.get("sinks"), dict) and config["sinks"]:
            sink_set = set(config["sinks"].keys())

        goal = Counter({t: 1 for t in sink_set})

        explorer = ReactionPathwayExplorer(net)
        explorer.find_forward(
            start=start,
            goal=goal,
            enforce_stoichiometry=True,
            infer_missing=infer_missing,
            max_depth=self.max_depth,
            max_pathways=self.max_pathways,
            allow_reuse=False,
            strategy="dfs",
            stop_on_goal=True,
        )

        raw_paths = explorer.pathways or []
        unique_paths = dedupe_pathways_by_canonical(net, raw_paths)

        if not unique_paths:
            msg = (
                f"No pathways found for motif '{name}' (raw_paths={len(raw_paths)}). "
                "Hint: check for unproduced reactants or missing sources."
            )
            raise NoPathwaysError(msg)

        print(f"  total raw: {len(raw_paths)}   unique canonical: {len(unique_paths)}")

        for i, p in enumerate(unique_paths[:show_n], 1):
            print(f"\n  Pathway #{i}:")
            pretty_print_pathway(net, p, show_original=True)
            inferred = replay_pathway_and_collect_inferred(net, p, start=start)
            if inferred:
                print("   inferred inflows:", dict(inferred))
            else:
                print("   inferred inflows: (none)")

        seqs = [
            " | ".join(
                net.reactions[r].canonical_raw or net.reactions[r].original_raw
                for r in p.reaction_ids
            )
            for p in unique_paths
        ]
        return {
            "name": name,
            "n_raw": len(raw_paths),
            "n_unique": len(unique_paths),
            "examples": seqs[:show_n],
            "net": net,
            "paths": unique_paths,
        }

    def run_all(
        self,
        motif_map: Dict[str, tuple],
        *,
        infer_missing: bool = True,
        write_csv: bool = True,
        csv_path: str = "motifs_summary.csv",
    ) -> List[Dict[str, Any]]:
        """
        Run enumeration for all motifs given in ``motif_map`` and optionally write CSV.

        :param motif_map: mapping name -> (reactions_list, config_dict)
        :param infer_missing: pass-through to enumerate_motif.
        :param write_csv: whether to write a CSV summary.
        :param csv_path: output file path for CSV.
        :returns: list of per-motif summary dicts (order preserves motif_map iteration).
        """
        import csv
        from pathlib import Path

        results: List[Dict[str, Any]] = []
        rows = []
        for name, (rxs, cfg) in motif_map.items():
            try:
                summary = self.enumerate_motif(
                    name, rxs, cfg, infer_missing=infer_missing
                )
            except NoPathwaysError:
                # fail-fast: re-raise
                raise
            except Exception as exc:  # pragma: no cover - external errors
                print(f"  ERROR enumerating {name}: {exc}")
                summary = {"name": name, "n_raw": 0, "n_unique": 0, "error": str(exc)}
            results.append(summary)

            if summary.get("n_unique", 0) > 0:
                rows.append(
                    {
                        "motif": name,
                        "n_raw": summary["n_raw"],
                        "n_unique": summary["n_unique"],
                        "example": summary["examples"][0],
                    }
                )
            else:
                rows.append(
                    {
                        "motif": name,
                        "n_raw": summary.get("n_raw", 0),
                        "n_unique": 0,
                        "example": summary.get("error", ""),
                    }
                )

        if write_csv:
            outp = Path(csv_path)
            with outp.open("w", newline="") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=["motif", "n_raw", "n_unique", "example"]
                )
                writer.writeheader()
                writer.writerows(rows)
            print("\nWrote summary CSV to", outp.resolve())

        return results
