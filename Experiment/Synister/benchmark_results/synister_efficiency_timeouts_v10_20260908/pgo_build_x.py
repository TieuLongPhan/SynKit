import hashlib,json,shutil,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent
D=R/"pgo_x"
D.mkdir(exist_ok=True)
data=D/"data"
data.mkdir(exist_ok=True)
source=R/"stage_x_source/synkit/Chem/Mapper/exact/native_distance.cpp"
obj=D/"native_distance.o"
common=["-std=c++17","-O3","-Wall","-Wextra","-fPIC","-march=native","-mtune=native"]
compiler=shutil.which("g++")
if sys.argv[1]=="generate":
    flags=[*common,f"-fprofile-generate={data}","-fprofile-update=atomic"]
    subprocess.run([compiler,*flags,"-c",str(source),"-o",str(obj)],check=True)
    subprocess.run([compiler,"-shared",f"-fprofile-generate={data}",str(obj),"-o",str(D/"train.so")],check=True)
    script=(R/"pgo_train.py").read_text().replace('ROOT/"pgo/train.so"','ROOT/"pgo_x/train.so"').replace("pgo_train","pgo_train_x")
    (R/"pgo_train_x.py").write_text(script)
else:
    profiles={str(p.relative_to(D)):hashlib.sha256(p.read_bytes()).hexdigest() for p in data.rglob("*.gcda")}
    assert profiles,"Training did not emit any profile data"
    flags=[*common,f"-fprofile-use={data}","-fprofile-correction","-Werror=missing-profile"]
    result=subprocess.run([compiler,*flags,"-c",str(source),"-o",str(obj)],check=True,capture_output=True,text=True)
    subprocess.run([compiler,"-shared",str(obj),"-o",str(D/"optimized.so")],check=True)
    provenance={"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                "compiler":compiler,"compiler_version":subprocess.run([compiler,"--version"],capture_output=True,text=True,check=True).stdout,
                "flags":flags+["-shared"],"profile_sha256":profiles,
                "resolved_target_options":subprocess.run([compiler,*common,"-Q","--help=target","-x","c++","-c","/dev/null","-o","/dev/null"],check=True,capture_output=True,text=True).stdout,
                "training_script_sha256":hashlib.sha256((R/"pgo_train_x.py").read_bytes()).hexdigest(),
                "compiler_stderr":result.stderr,
                "profile_scope":"branch/value frequencies only, no answer cache"}
    key=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    target=R/"build"/f"libsynkit_distance_{key[:20]}.so"
    if target.exists():
        assert target.read_bytes()==(D/"optimized.so").read_bytes(), 'Existing artifact differs'
    else:
        shutil.copyfile(D/"optimized.so",target)
    provenance.update(build_key=key,library=str(target),library_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    target.with_suffix(".json").write_text(json.dumps(provenance,indent=2)+"\n")
    (R/"stage_x_library.txt").write_text(str(target)+"\n")
    print(target)
