"""Internal CPU test worker, isolated from inherited pytest selection."""
import argparse
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    from multiscale_role.entry_guard import test_environment
    env=test_environment(os.environ);os.environ.clear();os.environ.update(env);os.chdir(ROOT)
    import pytest
    from multiscale_role.pytest_gate import FullSuiteGate
    return pytest.main(['-q','-c',str(ROOT/'pytest.ini'),'-o','addopts=',
        '--confcutdir='+str(ROOT/'tests'),'tests','--disable-warnings',
        '--junitxml='+str(Path(a.output)/'tests.xml')],plugins=[FullSuiteGate(ROOT,Path(a.output)/'test_execution.json')])


if __name__=='__main__':raise SystemExit(main())
