"""Package exact source and ALL non-weight evidence, after a complete audit."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

BASE="3bcbbd91545c24b32961bde52ac738262b9aa99d"
SOURCE="3a8d55234ccbd13859aed7f2d76fe17c5d0d4cf2"

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument("--results",type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parent;exp=root.parent
    run=a.results/"learning_run01"
    assert json.loads((run/"audit.json").read_text())["status"]=="passed"
    assert json.loads((run/"worker_receipt.json").read_text())["status"]=="completed"
    assert json.loads((run/"worker_receipt.json").read_text())["scientific_updates"]==36000
    ignore=shutil.ignore_patterns("__pycache__",".pytest_cache","*.pyc","*.pt")
    evidence=root/"evidence";shutil.copytree(a.results,evidence,ignore=ignore)
    inventory={str(p.relative_to(a.results)):dict(bytes=p.stat().st_size,sha256=sha(p),public_payload=False,
               reason="retained local model/optimizer/RNG state; hashes only in review artifact") for p in a.results.rglob("*.pt")}
    (evidence/"local_checkpoint_inventory.json").write_text(json.dumps(inventory,indent=2)+"\n")
    patch=subprocess.check_output(["git","diff",BASE,SOURCE,"--","experiments/role_robust_training_v1"],cwd=exp.parent)
    assert patch.startswith(b"diff --git")
    patchname="source_vs_3bcbbd9.patch";(root/patchname).write_bytes(patch)
    bundle=root/"review_bundle";bundle.mkdir()
    for name in ("role_robust_training_v1","role_specialization_v1","role_late_paired_v1","role_error_recovery_v1"):
        shutil.copytree(exp/name,bundle/name,ignore=ignore)
    prev=bundle/"role_condition_diagnosis_v1";prev.mkdir()
    shutil.copytree(exp/"role_condition_diagnosis_v1/baseline",prev/"baseline",ignore=ignore)
    for name in ("diagnose.py","late_fusion.py","test_late_fusion.py"):shutil.copy2(exp/"role_condition_diagnosis_v1"/name,prev/name)
    old=exp/"role_specialization_v1_results/evidence/learning_run01"
    dest=bundle/"role_specialization_v1_results/evidence/learning_run01";dest.mkdir(parents=True)
    for name in ("resolved_config.json","weight_hashes.json","source_manifest.json","environment.json"):shutil.copy2(old/name,dest/name)
    panel=bundle/"role_error_recovery_v1_results/evidence/archived_panel";panel.mkdir(parents=True)
    shutil.copy2(exp/"role_error_recovery_v1_results/evidence/archived_panel/panel.json",panel/"panel.json")
    shutil.copytree(evidence,bundle/"results")
    for name in ("README.md","RESULTS.md","analyze.py","plot_curves.py",patchname):shutil.copy2(root/name,bundle/name)
    manifest={str(p.relative_to(bundle)):sha(p) for p in sorted(bundle.rglob("*")) if p.is_file()}
    assert not any(n.endswith(".pt") for n in manifest)
    (bundle/"SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n,h in manifest.items()))
    path=root/"role_robust_training_6000_v1_review.zip"
    with zipfile.ZipFile(path,"x",zipfile.ZIP_DEFLATED) as z:
        for f in sorted(bundle.rglob("*")):
            if f.is_file():z.write(f,"role_robust_training_6000_v1_review/"+str(f.relative_to(bundle)))
    receipt=dict(sha256=sha(path),bytes=path.stat().st_size,payload_files=len(manifest),
                 scientific_source_commit=SOURCE,baseline=BASE,scientific_updates=36000,complete_chains=288,
                 private_checkpoint_files_retained_local=len(inventory))
    (root/"release_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    (root/"DELIVERY_SHA256SUMS").write_text(f"{sha(path)}  {path.name}\n{sha(root/patchname)}  {patchname}\n")
    print(json.dumps(receipt))

if __name__=="__main__":main()
