"""Archive every outcome and immutable source; no training/model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def copy_evidence(source,destination):
    destination.mkdir(parents=True,exist_ok=False)
    for p in sorted(source.rglob("*")):
        if p.is_file() and p.suffix!=".pt":
            q=destination/p.relative_to(source);q.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,q)

def main():
    ap=argparse.ArgumentParser()
    for flag in ("results","baseline","panel"):ap.add_argument("--"+flag,type=Path,required=True)
    a=ap.parse_args();root=Path(__file__).resolve().parent;exp=root.parent
    run=a.results/"learning_run01";ledger=json.loads((run/"activation.json").read_text())
    for name,status in (("audit.json","completed"),("joint_independent_audit.json","passed"),("input_audit.json","passed")):
        assert json.loads((run/name).read_text())["status"]==status
    assert time.time()<ledger["deadline_epoch"]
    evidence=root/"evidence";copy_evidence(a.results,evidence)
    p=evidence/"input_panel";p.mkdir()
    for name in ("panel.json","resolved_config.json"):shutil.copy2(a.panel/name,p/name)
    hashes=json.loads((run/"weight_hashes.json").read_text());assert len(hashes)==6
    for name,h in hashes.items():assert sha(run/name)==h
    (root/"weight_storage.json").write_text(json.dumps(dict(local_path=str(run),hashes=hashes,
        statement="All six actual synthetic initial/final weights retained locally; public archive contains hash receipts."),indent=2)+"\n")
    patch=subprocess.check_output(["git","diff","4a186b177448148cae765e16ad553e7aada7d29f","HEAD",
        "--","experiments/role_low_noise_v1"],cwd=exp.parent)
    assert len(patch)>1000
    (root/"source_vs_4a186b1.patch").write_bytes(patch)
    bundle=root/"review_bundle";bundle.mkdir(exist_ok=False)
    for name in ("role_low_noise_v1","role_degree_residual_v1","role_specialization_v1","role_late_paired_v1"):
        shutil.copytree(exp/name,bundle/"experiments"/name,ignore=shutil.ignore_patterns("__pycache__",".pytest_cache","*.pyc"))
    prev=bundle/"experiments/role_condition_diagnosis_v1";prev.mkdir()
    shutil.copytree(exp/"role_condition_diagnosis_v1/baseline",prev/"baseline",ignore=shutil.ignore_patterns("__pycache__",".pytest_cache","*.pyc"))
    for name in ("late_fusion.py","test_late_fusion.py","diagnose.py"):shutil.copy2(exp/"role_condition_diagnosis_v1"/name,prev/name)
    shutil.copytree(evidence,bundle/"results");copy_evidence(a.baseline,bundle/"baseline_evidence")
    for name in ("RESULTS.md","weight_storage.json","source_vs_4a186b1.patch"):shutil.copy2(root/name,bundle/name)
    tools=bundle/"experiments/role_low_noise_v1_results";tools.mkdir()
    for name in ("state_coverage.py","summarize.py","details.py"):shutil.copy2(root/name,tools/name)
    manifest={str(p.relative_to(bundle)):sha(p) for p in sorted(bundle.rglob("*")) if p.is_file()}
    (bundle/"SHA256SUMS").write_text("".join(f"{h}  {p}\n" for p,h in manifest.items()))
    archive=root/"role_low_noise_v1_review.zip"
    with zipfile.ZipFile(archive,"x",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(bundle.rglob("*")):
            if p.is_file():z.write(p,"role_low_noise_v1_review/"+str(p.relative_to(bundle)))
    receipt=dict(zip_sha256=sha(archive),bytes=archive.stat().st_size,payload_files=len(manifest),
        cumulative_seconds=time.time()-ledger["activation_epoch"],no_reset=True)
    assert receipt["cumulative_seconds"]<2700
    (root/"release_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))
if __name__=="__main__":main()
