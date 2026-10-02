import ast, collections, importlib.util, json, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
spec = importlib.util.spec_from_file_location("bench", REPO / "scripts/benchmark_synister_timeouts.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

class StripDocstrings(ast.NodeTransformer):
    def strip(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
        return node
    visit_Module = strip
    visit_FunctionDef = strip
    visit_AsyncFunctionDef = strip
    visit_ClassDef = strip

def code(path):
    return ast.dump(StripDocstrings().visit(ast.parse(path.read_text())), include_attributes=False)

textdiff = []
for path in sorted((REPO / "synkit").rglob("*.py")):
    frozen = ROOT / "component_source" / path.relative_to(REPO)
    assert code(path) == code(frozen), path
    if path.read_bytes() != frozen.read_bytes():
        textdiff.append(str(path.relative_to(REPO)))
selection = json.loads((ROOT / "remaining_selection.json").read_text())
records = [json.loads((ROOT / "component_remaining" / (str(t["source_line"]) + "_" + t["mode"] + ".json")).read_text()) for t in selection["tasks"]]
solved = [r for r in records if r["result"]["complete"]]
left = [r for r in records if not r["result"]["complete"]]
keys = {(r["source_line"], r["mode"]) for r in left}
remaining = dict(selection)
remaining["tasks"] = [t for t in selection["tasks"] if (t["source_line"], t["mode"]) in keys]
remaining["selection"] = "20 tasks still incomplete after component/twin symmetry enhancement; measured run component_remaining."
(ROOT / "remaining_after_enhancement.json").write_text(json.dumps(remaining, indent=2) + "\n")
report = dict(
    baseline_1200=json.loads((REPO / "benchmark_results/synister_efficiency_1200_timeouts_20260908/run/summary.json").read_text()),
    followup=json.loads((ROOT / "component_remaining/summary.json").read_text()),
    regression=json.loads((ROOT / "final_regression/regression_audit.json").read_text()),
    recovered_by_mode=dict(collections.Counter(r["mode"] for r in solved)),
    remaining_by_mode=dict(collections.Counter(r["mode"] for r in left)),
    remaining_by_phase=dict(collections.Counter(r["result"]["backend_statistics"]["search"]["phase"] for r in left)),
    recovered_structure_complete=sum(r["result"]["structure"]["complete"] for r in solved),
    recovered_quotient_complete=sum(r["result"]["symmetry_quotient_complete"] for r in solved),
    recovered_labeled_count_known=sum(r["result"]["labeled_solution_count"] is not None for r in solved),
    recovered_median_wall_seconds=statistics.median(r["wall_seconds"] for r in solved),
    recovered_max_wall_seconds=max(r["wall_seconds"] for r in solved),
    measured_implementation_sha256=bench.implementation_hash(ROOT / "component_source"),
    current_implementation_sha256=bench.implementation_hash(REPO),
    current_executable_ast_matches_measured=True,
    formatting_or_docstring_only_differences=textdiff,
    tests=dict(passed=181, skipped=1, known_provenance_deselected=1),
)
(ROOT / "completion_summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
