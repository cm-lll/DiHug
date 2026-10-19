"""Exact complete pytest collection, isolated filters, zero skips, CPU only."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from run import release_hash

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();start=time.monotonic()
    out=Path(a.output).resolve()
    if out.exists() or out==ROOT or ROOT in out.parents or out in ROOT.parents:raise ValueError('fresh disjoint evidence output required')
    release_hash();expected=json.loads((ROOT/'test_suite.json').read_text())['nodeids']
    env={k:v for k,v in os.environ.items() if not k.startswith('PYTEST_')}
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT))
    # Package-local node IDs in BOTH repository and stand-alone ZIP. Do not
    # inherit a parent repository's rootdir/config or silently weaken identity checks.
    cmd=[sys.executable,'-m','pytest','--rootdir',str(ROOT),'-c',os.devnull,'-p','no:cacheprovider','tests','-q']
    collected=subprocess.run(cmd+['--collect-only'],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60,check=True)
    ids=[s for s in collected.stdout.splitlines() if '::' in s and not s.startswith(' ')]
    if ids!=expected:raise ValueError('complete frozen test collection differs')
    out.mkdir(parents=True)
    result=subprocess.run(cmd+['--junitxml='+str(out/'tests.xml')],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    (out/'tests.log').write_text(result.stdout);tree=ET.parse(out/'tests.xml')
    cases=tree.findall('.//testcase')
    if result.returncode or len(cases)!=len(ids) or any(tree.findall('.//'+tag) for tag in ('failure','error','skipped')):raise ValueError('tests failed/missing/skipped')
    observed=[x.attrib['classname']+'.'+x.attrib['name'] for x in cases]
    intended=[s.split('::')[0].replace('/','.').removesuffix('.py')+'.'+'.'.join(s.split('::')[1:]) for s in ids]
    if observed!=intended:raise ValueError('executed test identities differ')
    data=dict(status='passed',cases=len(cases),skipped=0,exact_collection=True,wall_seconds=time.monotonic()-start,
        child_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
        peak_child_rss_mib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024,scope='small CPU engineering checks; not model quality')
    (out/'acceptance.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))


if __name__=='__main__':main()
