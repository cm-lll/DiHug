"""Independent unzip, complete payload hashes, public entry and full regressions."""
import hashlib
from collections import Counter
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
archive=root/"role_degree_residual_v1_review.zip"
destination=Path(tempfile.mkdtemp(prefix="role-degree-review-"));start=time.time()
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    z.extractall(destination)
package=destination/"role_degree_residual_v1_review";checked=0
for line in (package/"SHA256SUMS").read_text().splitlines():
    h,p=line.split("  ",1)
    assert hashlib.sha256((package/p).read_bytes()).hexdigest()==h;checked+=1
exp=package/"experiments";prev=exp/"role_condition_diagnosis_v1"
env=dict(os.environ,PYTEST_ADDOPTS="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="",
    PYTHONPATH=os.pathsep.join(map(str,[prev/"baseline",prev,exp/"role_specialization_v1"])))
commands=[
    [sys.executable,str(exp/"role_degree_residual_v1/learning.py"),"--help"],
    [sys.executable,str(exp/"role_degree_residual_v1/replay.py"),"--help"],
    [sys.executable,str(exp/"role_degree_residual_v1_results/fixed_input_response.py"),"--help"],
    [sys.executable,"-m","pytest","-q",str(prev/"baseline/tests"),str(prev/"test_late_fusion.py"),
     str(exp/"role_specialization_v1/test_capacity.py"),str(exp/"role_degree_residual_v1/test_residual.py"),
     "--junitxml="+str(root/"unpacked_tests.xml")]]
for i,command in enumerate(commands):
    remaining=ledger["deadline_epoch"]-time.time();assert remaining>0
    with (root/f"unpacked_{i}.log").open("w") as f:
        r=subprocess.run(command,cwd=package,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=remaining)
    assert r.returncode==0,(i,r.returncode)
tree=ET.parse(root/"unpacked_tests.xml")
assert len(tree.findall(".//testcase"))==94
assert not tree.findall(".//failure")+tree.findall(".//error")+tree.findall(".//skipped")
original=ET.parse(root/"evidence/full_tests.xml")
def identities(xml):
    return Counter((r.get("classname","").split(".")[-1],r.get("name")) for r in xml.findall(".//testcase"))
assert identities(tree)==identities(original),"unpacked test set differs from executed source set"
receipt=dict(status="passed",archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    files=checked,tests=94,failures=0,skipped=0,exact_test_set_verified=True,entry_help_passed=True,
    wall_seconds=time.time()-start,cumulative_seconds=time.time()-ledger["activation_epoch"],
    budget_seconds=2700,no_reset=True,independent_unpacked_directory=str(destination))
assert receipt["cumulative_seconds"]<2700
(root/"unpacked_acceptance.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
