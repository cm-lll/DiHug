"""Release-only frozen B1 collection; never changes native133 test contract."""
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent/'package';sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from multiscale_role.entry_guard import test_environment


def main():
    env=test_environment(os.environ);os.environ.clear();os.environ.update(env);os.chdir(ROOT)
    import pytest
    class Collector:
        ids=[]
        def pytest_collection_finish(self,session):self.ids=[i.nodeid for i in session.items]
    c=Collector();code=pytest.main(['--collect-only','-q','-p','no:cacheprovider','-c',str(ROOT/'pytest.ini'),'-o','addopts=',
                                  '--confcutdir='+str(ROOT/'b1_tests'),'b1_tests'],plugins=[c])
    if code!=0 or not c.ids or len(set(c.ids))!=len(c.ids):raise ValueError('B1 collection incomplete')
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'b1_tests').rglob('*.py')) if '__pycache__' not in p.parts}
    (ROOT/'B1_TEST_SUITE.json').write_text(json.dumps(dict(version='B1_attribution_v1',total=len(c.ids),nodeids=sorted(c.ids),test_file_sha256=files),indent=2)+'\n')


if __name__=='__main__':main()
