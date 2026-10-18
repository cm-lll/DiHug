import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import pytest
from test_multiscale import example
from multiscale_role.evaluation import degree_w1,ForwardPMFCache


def test_target_cache_outer_inner_tail_and_flags_cannot_mutate():
    l,pf,k,g,p,state=example();before=g.binding();loss=g.statistic(p,0)[0]
    cf,tail=g.target(0)
    with pytest.raises(TypeError):cf['0']=np.zeros(96)
    with pytest.raises(ValueError):cf['0'][0]=123
    with pytest.raises(ValueError):cf['0'].setflags(write=True)
    with pytest.raises(TypeError):tail['0']=(999.,)
    with pytest.raises(TypeError):tail['0'][0]=999.
    with pytest.raises(TypeError):g._target_cache[0]=({}, {})
    with pytest.raises(AttributeError):g._target_cache={}
    assert g.statistic(p,0)[0]==loss and g.binding()==before


def test_private_cache_corruption_detected_by_binding():
    l,pf,k,g,p,state=example();g.target(0)
    # Intentional private-internals tampering, beyond the public immutable API.
    g._MultiScaleGuide__target_cache[0]=({'0':np.zeros(96)}, {'0':(999.,)})
    with pytest.raises(ValueError,match='cached target changed'):g.binding()


def test_exact_forward_PMF_cache_equals_uncached_and_is_readonly():
    l,pf,k,g,p,state=example();cache=ForwardPMFCache()
    uncached=degree_w1(p,g,0)
    first=degree_w1(p,g,0,cache=cache);second=degree_w1(p,g,0,cache=cache)
    assert first==second==uncached and cache.hits>0
    values=next(iter(cache._items.values()))[0]
    with pytest.raises(ValueError):values.setflags(write=True)


def wait_for(path,timeout=5):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        if path.is_file():return
        time.sleep(.01)
    raise AssertionError('fixture did not become ready')


@pytest.mark.parametrize('action',['timeout','STOP','SIGINT','SIGTERM','SIGHUP','success','missing'])
def test_owned_process_group_cleanup_and_status(tmp_path,action):
    out=tmp_path/action
    fixture=Path(__file__).with_name('process_fixture.py')
    sentinel=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    parent=subprocess.Popen([sys.executable,str(fixture),action,str(out)])
    try:
        wait_for(out/'ready.json')
        pids=json.loads((out/'ready.json').read_text())
        if action=='STOP':(out/'STOP').touch()
        elif action.startswith('SIG'):os.kill(parent.pid,getattr(signal,action))
        result=parent.wait(timeout=9)
        record=json.loads((out/'controller.json').read_text())
        assert record['cleanup']==dict(direct_child_reaped=True,owned_group_gone=True)
        for pid in pids.values():assert not Path('/proc/'+str(pid)).exists()
        assert sentinel.poll() is None # no unrelated-process termination
        assert record['SIGKILL_catchable'] is False
        if action=='success':assert result==0 and record['status']=='completed'
        elif action=='missing':assert result!=0 and record['status']=='failed'
        elif action=='timeout':assert result==124 and record['status']=='incomplete'
        else:assert result!=0 and record['status']=='stopped' and record['reason']==action
        assert record['controller_wall_seconds']<7
    finally:
        if parent.poll() is None:parent.terminate();parent.wait(timeout=8)
        sentinel.terminate();sentinel.wait(timeout=3)
