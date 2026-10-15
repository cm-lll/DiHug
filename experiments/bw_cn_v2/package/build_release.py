"""Freeze independent CN experiment, exact v9 method preservation and diff."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import difflib

ROOT=Path(__file__).resolve().parent
V9SHA="fc70583c3d06e308486a4fe65ce3da8c3b73554268c685ca251300ee837b1ccf"


def sha(b):return hashlib.sha256(b).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument("--v9-zip",type=Path,required=True)
    p.add_argument("--prepare",action="store_true");p.add_argument("--output",type=Path)
    a=p.parse_args();assert sha(a.v9_zip.read_bytes())==V9SHA
    with zipfile.ZipFile(a.v9_zip) as z:old={n.split("/",1)[1]:z.read(n) for n in z.namelist()}
    files={}
    for folder in ("hsgd","data","evidence","tests"):
        for path in sorted((ROOT/folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in (".py",".json",".npz",""):
                files[str(path.relative_to(ROOT))]=path.read_bytes()
    for n in ("run.sh","说明.txt","build_release.py","pyg_inspector_compat.diff","local_acceptance.json","cn_v1_to_v2.diff"):
        if (ROOT/n).is_file():files[n]=(ROOT/n).read_bytes()
    # runtime.py has the reviewed CN legacy diagnostic readout correction.
    frozen=["hsgd/kernel.py","hsgd/noising.py","hsgd/slots.py","hsgd/vendor/conv_transformer_model.py",
        "hsgd/vendor/transconv_layer.py","hsgd/vendor/layers.py","hsgd/vendor/sample_edges_utils_typed.py",
        "hsgd/learning_validation.py","hsgd/bw_preflight/config.json","data/IMDB_movie_reference.npz",
        "data/IMDB_preprocessing_provenance.json"]
    for n in frozen:assert files[n]==old[n],n
    patch=[];changed=[]
    for n in sorted(files):
        if n.startswith("hsgd/") and n.endswith(".py") and files[n]!=old.get(n,b""):
            changed.append(n)
            patch.extend(difflib.unified_diff(old.get(n,b"").decode().splitlines(True),files[n].decode().splitlines(True),
                fromfile="a/"+n if n in old else "/dev/null",tofile="b/"+n))
    files["v9_to_cn_v2.diff"]="".join(patch).encode()
    (ROOT/"v9_to_cn_v2.diff").write_bytes(files["v9_to_cn_v2.diff"])
    manifest=dict(version="bw_cn_v2",source_commit=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
        v9_zip_sha256=V9SHA,method=dict(T=1000,query_blocks=20,default_max_updates=10000,
            arms=["baseline","cn"],cn_hidden=32,expected_new_IMDB_parameters=10116,
            shared_context="first existing Transformer layer on Xt alone, once per frozen context",
            feature_sum="MLP(h_w,type_w,onehot(r1)+onehot(r2),target_relation,t/T), sum over exact common neighbors",
            readout="bias-free linear(sum features,counts by middle type), residual edge-present logit",
            init="zero readout preserves baseline initial prediction",hard_constraints_added=False),
        byte_preserved=frozen,changes=changed,
        files={n:dict(sha256=sha(b),bytes=len(b)) for n,b in sorted(files.items())})
    (ROOT/"manifest_cn_v2.json").write_text(json.dumps(manifest,indent=2))
    if a.prepare:print("prepared exact source manifest; no model/training execution");return
    if not a.output or "local_acceptance.json" not in files:p.error("release needs acceptance and --output")
    files["manifest_cn_v2.json"]=(ROOT/"manifest_cn_v2.json").read_bytes()
    files["SHA256SUMS"]="".join(sha(b)+"  "+n+"\n" for n,b in sorted(files.items())).encode()
    a.output.mkdir(parents=True,exist_ok=True);archive=a.output/"dihug_sparse_bw_cn_v2.zip"
    with zipfile.ZipFile(archive,"x",zipfile.ZIP_DEFLATED) as z:
        for n,b in sorted(files.items()):z.writestr("dihug_sparse_bw_cn_v2/"+n,b)
    receipt=dict(archive=str(archive.resolve()),sha256=sha(archive.read_bytes()),bytes=archive.stat().st_size,
        files=len(files),source_commit=manifest["source_commit"],byte_preserved=frozen)
    (a.output/"release.json").write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))


if __name__=="__main__":main()
