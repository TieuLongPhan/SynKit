"""Check the added type guard on every frozen cohort input and audit its scope."""
import ast,csv,gzip,hashlib,json
from pathlib import Path
from synkit.Chem.Mapper import blinded_mapped_reaction_problem
from synkit.Chem.Mapper.exact.native_candidates import _validate_native_atom_types
R=Path(__file__).resolve().parent
selection=json.loads((R/"selection_1200.json").read_text())
dataset=Path(selection["dataset"])
assert hashlib.sha256(dataset.read_bytes()).hexdigest()==selection["dataset_sha256"]
with gzip.open(dataset,"rt") as stream:
    rows={int(row["source_line"]):row for row in csv.DictReader(stream)}
checked=set();types=set()
for task in selection["tasks"]:
    key=task["source_line"]
    row=rows[key]
    assert row["reaction_id"]==task["reaction_id"]
    if key in checked:continue
    reaction=row["mapped_reaction"]
    if "|" in reaction:
        reaction,identifier=reaction.rsplit("|",1)
        assert identifier==row["reaction_id"]
    problem=blinded_mapped_reaction_problem(reaction,heavy_only=True,blind_seed="synister-global-v1")
    a,b=(g.props["atomic numbers"] for g in problem.lgp)
    _validate_native_atom_types(a,b)
    types.update(type(v).__module__+"."+type(v).__qualname__ for v in (*a,*b))
    checked.add(key)
root=R.parents[1]
old=R/"native_source/synkit/Chem/Mapper/exact/native_candidates.py"
new=root/"synkit/Chem/Mapper/exact/native_candidates.py"
tree=ast.parse(new.read_text())
tree.body=[node for node in tree.body if not(isinstance(node,ast.FunctionDef) and node.name=="_validate_native_atom_types")]
for node in tree.body:
    if isinstance(node,ast.FunctionDef) and node.name=="prepare_native_candidates":
        node.body=[child for child in node.body if not(isinstance(child,ast.Expr) and isinstance(child.value,ast.Call) and isinstance(child.value.func,ast.Name) and child.value.func.id=="_validate_native_atom_types")]
same=ast.dump(tree)==ast.dump(ast.parse(old.read_text()))
assert same
package=R/"native_source/synkit"
changes=[str(p.relative_to(package)) for p in package.rglob("*") if p.suffix in (".py",".cpp") and p.read_bytes()!=(root/"synkit"/p.relative_to(package)).read_bytes()]
assert changes==["Chem/Mapper/exact/native_candidates.py"]
summary={"tasks_validated":len(selection["tasks"]),"unique_reactions_validated":len(checked),"atom_label_types":sorted(types),"guard_rejections":0,"only_post_freeze_changed_source_files":changes,"computation_ast_unchanged_after_removing_guard":same,"frozen_wrapper_sha256":hashlib.sha256(old.read_bytes()).hexdigest(),"current_wrapper_sha256":hashlib.sha256(new.read_bytes()).hexdigest(),"timing_scope":"Cohort timings use the frozen pre-guard wrapper. The added input-check overhead is not included. Every cohort input passes the guard, and the remaining computation is identical."}
(R/"post_freeze_type_guard_audit.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary))
