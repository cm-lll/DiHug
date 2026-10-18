"""Owned synthetic process tree; no scientific/model work."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


if __name__=='__main__':
    mode=sys.argv[1];out=Path(sys.argv[2])
    if mode=='grandchild':
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        (out/'grandchild.json').write_text(json.dumps(dict(pid=os.getpid())))
        while True:time.sleep(.05)
    elif mode=='child':
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        grand=subprocess.Popen([sys.executable,__file__,'grandchild',str(out)])
        until=time.monotonic()+3
        while not (out/'grandchild.json').exists() and time.monotonic()<until:time.sleep(.01)
        (out/'ready.json').write_text(json.dumps(dict(pid=os.getpid(),grandchild=grand.pid)))
        if sys.argv[3]=='exit':(out/'result.ok').write_text('fixture');raise SystemExit(0)
        while True:time.sleep(.05)
    else:
        from multiscale_role.runtime_guard import supervise
        out.mkdir(parents=True,exist_ok=False)
        child_mode='exit' if mode in ('success','missing') else 'wait'
        def validate(check):
            check()
            if mode=='missing':raise ValueError('missing mandatory fixture evidence')
            if not (out/'result.ok').is_file():raise ValueError('no result evidence')
        raise SystemExit(supervise([sys.executable,__file__,'child',str(out),child_mode],out,7.,validator=validate))
