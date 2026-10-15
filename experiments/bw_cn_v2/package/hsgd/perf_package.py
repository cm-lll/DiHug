"""Explicit v5 packager; snapshot v4 remains unchanged and separately bound."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
from hsgd.bw_preflight.common import PACKAGE, sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--v4-archive",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    archive=args.output/"dihug_sparse_bw_perf_v5.zip"
    if archive.exists():raise RuntimeError("never overwrite an existing archive")
    expected_v4="8e3f1805deeecaafcf14aecb5da39ccc573cf09c43cdcb093ba9d98f47d9653f"
    if sha(args.v4_archive)!=expected_v4:raise RuntimeError("original v4 archive hash differs")
    with zipfile.ZipFile(args.v4_archive) as z:
        old={n.split("/",1)[1]:z.read(n) for n in z.namelist()}
    files={}
    for folder in ("hsgd","tests","evidence"):
        for path in sorted((PACKAGE/folder).rglob("*")):
            if path.is_file() and path.suffix in (".py",".json",".yaml") or path.is_file() and path.name=="LICENSE":
                files[str(path.relative_to(PACKAGE))]=path.read_bytes()
    files["data/IMDB_movie_reference.npz"]=(PACKAGE/"data/IMDB_movie_reference.npz").read_bytes()
    files["pyg_inspector_compat.diff"]=old["pyg_inspector_compat.diff"]
    assert files["pyg_inspector_compat.diff"] and b"@@" in files["pyg_inspector_compat.diff"]
    files["manifest_v4_source.json"]=old["manifest_v4.json"]
    files["local_acceptance.json"]=(PACKAGE/"results/local_acceptance/acceptance.json").read_bytes()
    files["local_graph_cpu.json"]=(PACKAGE/"results/local_graph_cpu_final/summary.json").read_bytes()
    changed=[];patch=[]
    for name in sorted(n for n in files if n.startswith("hsgd/")):
        previous=old.get(name,b"");current=files[name]
        if previous!=current:
            changed.append(name)
            patch.extend(difflib.unified_diff(previous.decode().splitlines(True),current.decode().splitlines(True),
                                             fromfile="a/"+name,tofile="b/"+name))
    files["v4_to_perf_v5.diff"]="".join(patch).encode()
    manifest=dict(version="bw_perf_v5",base_v4_archive_sha256=expected_v4,
        source_commit=subprocess.check_output(["git","-C",str(PACKAGE),"rev-parse","HEAD"],text=True).strip(),
        changed_or_added_hsgd_files=changed,
        scope="bounded performance/equivalence; no formal training/full chains; v4 forward and mathematics retained",
        weight_import="explicit pristine v4 source/B100 receipt required; read-only; not old-chain resume",
        files={n:dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)) for n,b in sorted(files.items())})
    files["manifest_perf_v5.json"]=json.dumps(manifest,indent=2).encode()
    files["SHA256SUMS"]="".join(hashlib.sha256(b).hexdigest()+"  "+n+"\n" for n,b in sorted(files.items())).encode()
    with zipfile.ZipFile(archive,"x",zipfile.ZIP_DEFLATED) as z:
        for n,b in sorted(files.items()):z.writestr("dihug_sparse_bw_perf_v5/"+n,b)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    (args.output/"manifest_perf_v5.json").write_bytes(files["manifest_perf_v5.json"])
    (args.output/"SHA256SUMS").write_bytes(files["SHA256SUMS"]+f"{sha(archive)}  {archive.name}\n".encode())
    print(json.dumps(dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
                         source_commit=manifest["source_commit"],files=len(files),changed_hsgd=changed),indent=2))


if __name__=="__main__":main()
