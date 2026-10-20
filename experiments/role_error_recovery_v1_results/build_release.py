"""Build review bundle, never run a scientific command."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results",type=Path,required=True)
    ap.add_argument("--panel",type=Path,required=True)
    a=ap.parse_args();root=Path(__file__).resolve().parent;exp=root.parent
    assert json.loads((a.results/"diagnostic_run01/audit.json").read_text())["status"]=="passed"
    evidence=root/"evidence";shutil.copytree(a.results,evidence)
    (evidence/"archived_panel").mkdir()
    shutil.copy2(a.panel/"panel.json",evidence/"archived_panel/panel.json")
    patch=subprocess.check_output(["git","diff","b8e97cc42e6e835e0ea228a6666e3a22a200e035","HEAD","--",
        "experiments/role_error_recovery_v1"],cwd=exp.parent)
    assert patch.startswith(b"diff --git")
    (root/"source_vs_b8e97cc.patch").write_bytes(patch)
    bundle=root/"review_bundle";bundle.mkdir()
    ignore=shutil.ignore_patterns("__pycache__",".pytest_cache","*.pyc")
    for name in ("role_error_recovery_v1","role_specialization_v1","role_late_paired_v1"):
        shutil.copytree(exp/name,bundle/name,ignore=ignore)
    prev=bundle/"role_condition_diagnosis_v1";prev.mkdir()
    shutil.copytree(exp/"role_condition_diagnosis_v1/baseline",prev/"baseline",ignore=ignore)
    for name in ("diagnose.py","late_fusion.py","test_late_fusion.py"):
        shutil.copy2(exp/"role_condition_diagnosis_v1"/name,prev/name)
    old=exp/"role_specialization_v1_results/evidence/learning_run01"
    new=bundle/"role_specialization_v1_results/evidence/learning_run01";new.mkdir(parents=True)
    names=["resolved_config.json","weight_hashes.json","source_manifest.json","environment.json","baseline_sources.json"]
    names += [f"seed{seed}/capacity_C_validation_1000.json" for seed in (2026102011,2026102012,2026102013)]
    for name in names:
        p=new/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/name,p)
    tools=bundle/"role_error_recovery_v1_results";tools.mkdir()
    for name in ("audit_and_summarize.py","report.py","RESULTS.md","source_vs_b8e97cc.patch"):
        shutil.copy2(root/name,tools/name)
    shutil.copytree(evidence,bundle/"results")
    manifest={str(p.relative_to(bundle)):sha(p) for p in sorted(bundle.rglob("*")) if p.is_file()}
    (bundle/"SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n,h in manifest.items()))
    archive=root/"role_error_recovery_v1_review.zip"
    with zipfile.ZipFile(archive,"x",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(bundle.rglob("*")):
            if p.is_file():z.write(p,"role_error_recovery_v1_review/"+str(p.relative_to(bundle)))
    receipt=dict(sha256=sha(archive),bytes=archive.stat().st_size,payload_files=len(manifest),
                 source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=exp.parent,text=True).strip())
    (root/"release_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    (root/"DELIVERY_SHA256SUMS").write_text(f"{sha(archive)}  {archive.name}\n{sha(root/'source_vs_b8e97cc.patch')}  source_vs_b8e97cc.patch\n")
    print(json.dumps(receipt))
if __name__=="__main__":main()
