import ast,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
old=R.parent/"synister_enhancement_v14_20260909/source"
new=R/"source"
class DefaultPath(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        self.generic_visit(node)
        if node.name=="as_dict":
            kept=[(a,d) for a,d in zip(node.args.kwonlyargs,node.args.kw_defaults) if a.arg!="copy_sequences"]
            node.args.kwonlyargs=[a for a,d in kept];node.args.kw_defaults=[d for a,d in kept]
        return node
    def visit_IfExp(self,node):
        self.generic_visit(node)
        if isinstance(node.test,ast.Name) and node.test.id=="copy_sequences":return node.body
        return node
    def visit_Call(self,node):
        self.generic_visit(node)
        node.keywords=[k for k in node.keywords if k.arg!="copy_sequences"]
        return node
differences=[]
for p in sorted((old/"synkit").rglob("*")):
    if p.suffix not in (".py",".cpp"):continue
    rel=p.relative_to(old); q=new/rel
    if p.read_bytes()==q.read_bytes():continue
    assert rel.as_posix() in ("synkit/Chem/Mapper/analysis.py","synkit/Chem/Mapper/spectrum.py"),rel
    a=ast.parse(p.read_text());b=DefaultPath().visit(ast.parse(q.read_text()))
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),rel
    differences.append(rel.as_posix())
assert len(differences)==2
report={"changed_package_files":differences,"default_path_ast_identical":True,
        "all_other_python_cpp_files_identical":True,
        "scope":"optional immutable-sequence JSON path only; no solver/canonicalization changes"}
(R/"source_scope_audit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
