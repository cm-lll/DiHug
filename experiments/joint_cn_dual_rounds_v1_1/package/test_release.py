"""Full collection validation, isolating inherited pytest selection settings."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dual.data import ROOT
from dual.io import verify_release,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('test evidence output exists')
    from dual.io import protect
    out=protect(a.output);out.mkdir(parents=True)
    verify_release()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    for name in ('PYTEST_ADDOPTS','PYTEST_PLUGINS'):env.pop(name,None)
    args=[sys.executable,'-m','pytest','-c','/dev/null','--rootdir='+str(ROOT),'-o','addopts=',
          '-o','junit_family=xunit1','-p','no:cacheprovider']
    start=time.monotonic()
    collect=subprocess.run(args+['--collect-only','-q','tests'],cwd=ROOT,env=env,capture_output=True,text=True)
    (out/'collection.txt').write_text(collect.stdout+collect.stderr)
    ids=sorted(x.strip() for x in collect.stdout.splitlines() if '::' in x and x.startswith('tests/'))
    expected=json.loads((ROOT/'full_expected_tests_v11.json').read_text())
    if collect.returncode or not ids or ids!=expected:raise ValueError('complete test collection differs from frozen nonempty IDs')
    test=subprocess.run(args+['-q','tests','--junitxml='+str(out/'tests.xml')],cwd=ROOT,env=env,capture_output=True,text=True)
    (out/'pytest.txt').write_text(test.stdout+test.stderr)
    cases=list(ET.parse(out/'tests.xml').iter('testcase'))
    bad=[x for x in cases if any(x.find(tag) is not None for tag in ('failure','error','skipped'))]
    ok=test.returncode==0 and len(cases)==len(expected) and not bad
    from hsgd.bw_preflight.common import environment
    atomic_json(out/'results.json',dict(status='passed' if ok else 'failed',exit_code=test.returncode,
        collected=len(ids),executed=len(cases),failed_or_skipped=len(bad),wall_seconds=time.monotonic()-start,
        child_peak_rss_mib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024,
        child_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
        environment=environment('cpu'),GPU_tests=0,formal_training_updates=0,scientific_graphs=0,
        test_scope='original core plus small CPU engineering fixtures, no real-data network forward',
        selection_environment_isolated=True))
    print((out/'results.json').read_text(),flush=True)
    return 0 if ok else 1

if __name__=='__main__':raise SystemExit(main())
