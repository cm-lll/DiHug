"""v6 archive with unchanged method config and preserved v5 base hash."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
from hsgd.bw_preflight.common import PACKAGE,sha


def main():
    p=argparse.ArgumentParser();p.add_argument("--v5-archive",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--acceptance",type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    expected="804a8d8176fe222eac088f6bdcb6e3c4ac6fc6ef24b9ca38add116ff50bf7606"
    if sha(args.v5_archive)!=expected:raise RuntimeError("original v5 archive differs")
    archive=args.output/"dihug_sparse_bw_runtime_v6.zip"
    if archive.exists():raise RuntimeError("existing archive protected; choose new output")
    with zipfile.ZipFile(args.v5_archive) as z:
        old={n.split("/",1)[1]:z.read(n) for n in z.namelist()}
    files={}
    for folder in ("hsgd","tests","evidence"):
        for path in sorted((PACKAGE/folder).rglob("*")):
            if path.is_file() and (path.suffix in (".py",".json",".yaml") or path.name=="LICENSE"):
                files[str(path.relative_to(PACKAGE))]=path.read_bytes()
    for name in ("data/IMDB_movie_reference.npz","pyg_inspector_compat.diff"):
        files[name]=(PACKAGE/name).read_bytes()
    assert files["pyg_inspector_compat.diff"]==old["pyg_inspector_compat.diff"] and len(files["pyg_inspector_compat.diff"])>0
    assert files["hsgd/bw_preflight/config.json"]==old["hsgd/bw_preflight/config.json"]
    # Preserve all old tests byte-for-byte; include full upstream provenance.
    for name,data in old.items():
        if name.startswith("tests/"):assert files[name]==data,name
    files["manifest_perf_v5_source.json"]=old["manifest_perf_v5.json"]
    files["local_acceptance_v6.json"]=args.acceptance.read_bytes()
    changes=[];patch=[]
    for name in sorted(n for n in files if n.startswith(("hsgd/","tests/"))):
        if files[name]!=old.get(name,b""):
            changes.append(name)
            patch.extend(difflib.unified_diff(old.get(name,b"").decode().splitlines(True),files[name].decode().splitlines(True),
                                            fromfile="a/"+name,tofile="b/"+name))
    files["v5_to_runtime_v6.diff"]="".join(patch).encode()
    manifest=dict(version="bw_runtime_v6",base_v5_archive_sha256=expected,
        source_commit=subprocess.check_output(["git","-C",str(PACKAGE),"rev-parse","HEAD"],text=True).strip(),
        graph_backend_default="upstream",method_config_changed=False,old_tests_preserved=True,
        scope="production performance integration plus finite engineering only; no formal training/full IMDB chains",
        changed_or_added_files=changes,files={n:dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)) for n,b in sorted(files.items())})
    files["manifest_runtime_v6.json"]=json.dumps(manifest,indent=2).encode()
    files["SHA256SUMS"]="".join(hashlib.sha256(b).hexdigest()+"  "+n+"\n" for n,b in sorted(files.items())).encode()
    with zipfile.ZipFile(archive,"x",zipfile.ZIP_DEFLATED) as z:
        for n,b in sorted(files.items()):z.writestr("dihug_sparse_bw_runtime_v6/"+n,b)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    (args.output/"SHA256SUMS").write_bytes(files["SHA256SUMS"]+f"{sha(archive)}  {archive.name}\n".encode())
    (args.output/"manifest_runtime_v6.json").write_bytes(files["manifest_runtime_v6.json"])
    print(json.dumps(dict(archive=str(archive),sha256=sha(archive),source_commit=manifest["source_commit"],
                         bytes=archive.stat().st_size,files=len(files),changes=changes),indent=2))


if __name__=="__main__":main()
