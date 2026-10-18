"""Release-time collection only; never auto-regenerate the runtime contract."""
import hashlib
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent/'package';sys.path.insert(0,str(ROOT))
from multiscale_role.entry_guard import test_environment


def main():
    env=test_environment(os.environ);os.environ.clear();os.environ.update(env);os.chdir(ROOT)
    import pytest
    class Collection:
        ids=[]
        def pytest_collection_finish(self,session):self.ids=[i.nodeid for i in session.items]
    c=Collection()
    code=pytest.main(['--collect-only','-q','-p','no:cacheprovider','-c',str(ROOT/'pytest.ini'),
                     '-o','addopts=','--confcutdir='+str(ROOT/'tests'),'tests'],plugins=[c])
    if code!=0 or len(c.ids)!=len(set(c.ids)) or len(c.ids)<=112:raise ValueError('incomplete release test collection')
    test_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((ROOT/'tests').rglob('*.py')) if '__pycache__' not in p.parts}
    spec=dict(version='bw_role_multiscale_tail_v1_2',total=len(c.ids),nodeids=sorted(c.ids),test_file_sha256=test_files,
              inherited_v1_1_cases=112,collection_mode='isolated environment; explicit config/addopts; plugin autoload disabled')
    (ROOT/'TEST_SUITE.json').write_text(json.dumps(spec,indent=2)+'\n')
    print('FROZEN FULL COLLECTION',len(c.ids))


if __name__=='__main__':main()
