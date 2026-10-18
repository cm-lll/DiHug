#!/usr/bin/env python3
"""Verify release bytes AND exact frozen test collection, zero skips required."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    start=time.monotonic();out=Path(a.output).resolve()
    if out==ROOT or ROOT in out.parents or out in ROOT.parents:raise ValueError('output must not overlap deployment')
    if out.exists():raise FileExistsError('fresh test evidence directory required')
    for line in (ROOT/'DYNAMIC_SHA256SUMS').read_text().splitlines():
        h,n=line.split('  ',1)
        if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h:raise ValueError('release hash differs: '+n)
    frozen=json.loads((ROOT/'dynamic_test_suite.json').read_text());expected=frozen['nodeids']
    env={k:v for k,v in os.environ.items() if not k.startswith('PYTEST_')}
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONDONTWRITEBYTECODE='1',
               CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(ROOT))
    command=[sys.executable,'-m','pytest','tests','matching_tests','-q']
    def run(extra):
        left=300-(time.monotonic()-start)
        if left<=0:raise TimeoutError('300 second local regression bound')
        return subprocess.run(command+extra,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=left)
    collected=run(['--collect-only']);nodeids=[s for s in collected.stdout.splitlines() if '::' in s and not s.startswith(' ')]
    if collected.returncode or nodeids!=expected:raise ValueError('actual full test collection differs from frozen release')
    out.mkdir(parents=True)
    test=run(['--junitxml='+str(out/'tests.xml')]);(out/'tests.log').write_text(test.stdout)
    if test.returncode:raise ValueError('CPU regression failed; inspect tests.log')
    tree=ET.parse(out/'tests.xml');cases=tree.findall('.//testcase')
    if len(cases)!=len(expected) or tree.findall('.//skipped') or tree.findall('.//failure') or tree.findall('.//error'):
        raise ValueError('missing/failed/skipped tests cannot pass')
    observed=[x.attrib['classname']+'.'+x.attrib['name'] for x in cases]
    identities=[s.split('::')[0].replace('/','.').removesuffix('.py')+'.'+'.'.join(s.split('::')[1:]) for s in expected]
    if observed!=identities:raise ValueError('executed test identities differ')
    result=dict(status='passed',cases=len(cases),skipped=0,exact_collection=True,environment_filters_isolated=True,
        cpu_child_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
        peak_child_rss_mib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024,
        wall_seconds=time.monotonic()-start,scope='CPU engineering tests only, no actual frozen6/BW verification')
    (out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
