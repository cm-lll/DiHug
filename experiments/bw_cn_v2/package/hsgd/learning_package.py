"""Independent v7 deployment; preserve byte-frozen v6 archive and tests."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
from hsgd.bw_preflight.common import PACKAGE,sha


def main():
    p=argparse.ArgumentParser();p.add_argument("--v6-archive",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--acceptance",type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    expected="aaa691344d5d5ade6cf29a761d5ec477f65d6bc737441052d6249f1679faf2fa"
    if sha(args.v6_archive)!=expected:raise RuntimeError("frozen v6 archive differs")
    archive=args.output/"dihug_sparse_bw_learning_v7.zip"
    if archive.exists():raise RuntimeError("archive protected; choose new output")
    with zipfile.ZipFile(args.v6_archive) as z:old={n.split("/",1)[1]:z.read(n) for n in z.namelist()}
    files={}
    for folder in ("hsgd","tests","evidence"):
        for path in sorted((PACKAGE/folder).rglob("*")):
            if path.is_file() and (path.suffix in (".py",".json",".yaml") or path.name=="LICENSE"):
                files[str(path.relative_to(PACKAGE))]=path.read_bytes()
    for name in ("data/IMDB_movie_reference.npz","data/IMDB_preprocessing_provenance.json","pyg_inspector_compat.diff"):
        files[name]=(PACKAGE/name).read_bytes()
    for name,data in old.items():
        if name.startswith("tests/"):assert files[name]==data,name
    assert files["hsgd/bw_preflight/config.json"]==old["hsgd/bw_preflight/config.json"]
    files["manifest_runtime_v6_source.json"]=old["manifest_runtime_v6.json"]
    files["local_acceptance_v7.json"]=args.acceptance.read_bytes()
    patch=[];changes=[]
    for name in sorted(n for n in files if n.startswith(("hsgd/","tests/"))):
        if files[name]!=old.get(name,b""):
            changes.append(name);patch.extend(difflib.unified_diff(old.get(name,b"").decode().splitlines(True),files[name].decode().splitlines(True),fromfile="a/"+name,tofile="b/"+name))
    files["v6_to_learning_v7.diff"]="".join(patch).encode()
    manifest=dict(version="bw_learning_v7",base_v6_archive_sha256=expected,
        source_commit=subprocess.check_output(["git","-C",str(PACKAGE),"rev-parse","HEAD"],text=True).strip(),
        method_config_changed=False,default_entry="hsgd.learning_launch",formal_BW_wall_seconds=28800,
        full_training_update_cutoff=None,sampling_gate_required=False,automatic_generation=False,
        files={n:dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)) for n,b in sorted(files.items())},changes=changes)
    files["manifest_learning_v7.json"]=json.dumps(manifest,indent=2).encode()
    files["SHA256SUMS"]="".join(hashlib.sha256(b).hexdigest()+"  "+n+"\n" for n,b in sorted(files.items())).encode()
    with zipfile.ZipFile(archive,"x",zipfile.ZIP_DEFLATED) as z:
        for name,b in sorted(files.items()):z.writestr("dihug_sparse_bw_learning_v7/"+name,b)
    (args.output/"manifest_learning_v7.json").write_bytes(files["manifest_learning_v7.json"])
    (args.output/"SHA256SUMS").write_bytes(files["SHA256SUMS"]+f"{sha(archive)}  {archive.name}\n".encode())
    print(json.dumps(dict(archive=str(archive),sha256=sha(archive),source_commit=manifest["source_commit"],bytes=archive.stat().st_size,files=len(files),changes=changes),indent=2))


if __name__=="__main__":main()
