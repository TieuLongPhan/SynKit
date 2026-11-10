"""Apply typed DPO rules from the Python CRN controller."""

from __future__ import annotations

from typing import Any


class DPOWorker:
    """Cache tuple rules and a native handle for one network build."""

    def __init__(self, rules, *, backend, library_path=None, radical_policy="strict", component_policy="union"):
        from synkit.Rule import SynRule

        self.backend = backend
        self.radical_policy = radical_policy
        self.component_policy = component_policy
        self.library = None
        if backend == "cpp":
            from synkit.Synthesis.Reactor._native.runtime import load

            self.library = load(library_path)
        self.rules = []
        for rule in rules:
            if isinstance(rule, str):
                rule = SynRule.from_smart(rule, format="tuple", implicit_h=True)
            if not isinstance(rule, SynRule) or rule._format != "tuple":
                raise ValueError("DPO CRN construction requires reaction strings or tuple-format SynRule objects")
            self.rules.append(rule)
        self.calls = self.candidates = self.applications = self.elapsed_ns = 0
        self.last_statistics = None
        self.component_rejected = 0

    @staticmethod
    def distinct_components(engine, mapping):
        """Require different rule components to use different host molecules."""
        import networkx as nx

        host = engine._matching_host_graph()
        owners = {node: index for index, component in enumerate(nx.connected_components(host)) for node in component}
        selected = []
        for component in nx.connected_components(engine.rule.left.raw):
            targets = {owners[mapping[node]] for node in component if node in mapping}
            if len(targets) != 1 or any(node not in mapping for node in component):
                return False
            selected.append(next(iter(targets)))
        return len(set(selected)) == len(selected) and set(selected) == set(owners.values())

    def __call__(self, task):
        from synkit.Synthesis.Reactor import WildcardEngine

        index, _rule, substrate, _explicit_h, _implicit_temp, strategy, keys = task
        engine = WildcardEngine(
            substrate,
            self.rules[index],
            backend=self.backend,
            library_path=str(self.library[1]) if self.library is not None else None,
            partial=False,
            explicit_h=False,
            implicit_temp=True,
            radical_policy=self.radical_policy,
            strategy=strategy or "all",
            automorphism=True,
        )
        if self.library is not None:
            # This private handle was verified once when the build began.
            engine._preloaded_native_library = self.library
        if self.component_policy == "distinct":
            if self.backend == "cpp":
                population = engine._native_result()
                keep = [
                    i for i, mapping in enumerate(population["mappings"]) if self.distinct_components(engine, mapping)
                ]
                self.component_rejected += len(population["mappings"]) - len(keep)
                engine._native_cache = {
                    **population,
                    "mappings": [population["mappings"][i] for i in keep],
                    "graphs": [population["graphs"][i] for i in keep],
                }
            else:
                mappings = engine.mappings
                engine._mappings = [m for m in mappings if self.distinct_components(engine, m)]
                self.component_rejected += len(mappings) - len(engine._mappings)
        products = list(dict.fromkeys(s.strip() for s in engine.smiles_list if s.strip()))
        self.calls += 1
        if self.backend == "cpp":
            stats = engine.native_statistics
            self.last_statistics = stats
            self.candidates += stats["candidates"]
            self.applications += stats["applications"]
            self.elapsed_ns += stats["elapsed_ns"]
        return index, keys, products

    def report(self) -> dict[str, Any]:
        return {
            "engine": "synkit_dpo_cpp" if self.backend == "cpp" else "synkit_dpo_python",
            "controller": "python",
            "kernel": self.backend,
            "reactor_library": str(self.library[1]) if self.library is not None else None,
            "radical_policy": self.radical_policy,
            "template_format": "tuple",
            "component_policy": self.component_policy,
            "component_rejected_applications": self.component_rejected,
            "hydrogen_model": "implicit",
            "partial": False,
            "rules_compiled": len(self.rules),
            "rule_calls": self.calls,
            "native_candidates": self.candidates if self.backend == "cpp" else None,
            "native_applications": self.applications if self.backend == "cpp" else None,
            "native_elapsed_ns": self.elapsed_ns if self.backend == "cpp" else None,
        }


# Each process retains its own compiled rules and native handle.
_POOL_WORKER = None


def initialize_dpo_worker(rules, backend, library_path, radical_policy, component_policy="union"):
    global _POOL_WORKER
    _POOL_WORKER = DPOWorker(
        rules,
        backend=backend,
        library_path=library_path,
        radical_policy=radical_policy,
        component_policy=component_policy,
    )


def apply_dpo_pool_task(task):
    before = _POOL_WORKER.component_rejected
    result = _POOL_WORKER(task)
    return result, _POOL_WORKER.last_statistics, _POOL_WORKER.component_rejected - before
