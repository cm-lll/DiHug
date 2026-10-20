import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile

root=Path(__file__).resolve().parent
ledger=json.loads((root/"evidence/learning_run01/activation.json").read_text())
archive=root/"role_specialization_v1_review.zip";destination=Path(tempfile.mkdtemp(prefix="role-capacity-review-"))
start=time.time()
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    z.extractall(destination)
package=destination/"role_specialization_v1_review";checked=0
for line in (package/"SHA256SUMS").read_text().splitlines():
    h,p=line.split("  ",1)
    assert hashlib.sha256((package/p).read_bytes()).hexdigest()==h;checked+=1
env=dict(os.environ,PYTEST_ADDOPTS="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="",
         PYTHONPATH=str(package/"role_condition_diagnosis_v1/baseline"))
commands=[
    [sys.executable,str(package/"role_specialization_v1/panel.py"),"--help"],
    [sys.executable,"-m","pytest","-q",str(package/"role_condition_diagnosis_v1/baseline/tests"),
     str(package/"role_condition_diagnosis_v1/test_late_fusion.py"),
     str(package/"role_specialization_v1/test_capacity.py"),"--junitxml="+str(root/"unpacked_tests.xml")]]
for i,command in enumerate(commands):
    remaining=ledger["deadline_epoch"]-time.time();assert remaining>0
    with (root/f"unpacked_{i}.log").open("w") as log:
        r=subprocess.run(command,cwd=package,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
    assert r.returncode==0
tree=ET.parse(root/"unpacked_tests.xml")
assert len(tree.findall(".//testcase"))==89
assert not tree.findall(".//failure")+tree.findall(".//error")+tree.findall(".//skipped")
receipt=dict(status="passed",archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=checked,
    tests=89,failures=0,skipped=0,entry_help_passed=True,wall_seconds=time.time()-start,
    cumulative_seconds=time.time()-ledger["activation_epoch"],budget_seconds=2700,no_reset=True,
    independent_unpacked_directory=str(destination))
assert receipt["cumulative_seconds"]<2700
(root/"unpacked_acceptance.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
