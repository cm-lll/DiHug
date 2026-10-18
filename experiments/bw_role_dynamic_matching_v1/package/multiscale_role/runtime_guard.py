"""Owned-process cleanup and bounded, STOP-aware I/O; stdlib only."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


class Budget:
    def __init__(self,deadline,stop,cleanup_deadline=None):
        self.deadline=deadline;self.stop=Path(stop);self.cleanup_deadline=cleanup_deadline
        self.signals=[];self.old_handlers={}
    def capture(self,signum,frame):self.signals.append(signum)
    def install(self):
        for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):self.old_handlers[s]=signal.signal(s,self.capture)
    def restore(self):
        for s,h in self.old_handlers.items():signal.signal(s,h)
        self.old_handlers.clear()
    def check(self):
        if self.signals:raise InterruptedError('worker received '+signal.Signals(self.signals[0]).name)
        if self.stop.exists():raise InterruptedError('safe STOP requested')
        if time.monotonic()>=self.deadline:raise TimeoutError('bounded work deadline reached')
    def cleanup_check(self):
        if self.cleanup_deadline is not None and time.monotonic()>=self.cleanup_deadline:
            raise TimeoutError('cleanup/save deadline reached')


def checked_sha(path,check=lambda:None):
    h=hashlib.sha256();check()
    with Path(path).open('rb') as f:
        while True:
            check();data=f.read(1024*1024)
            if not data:break
            h.update(data)
    check();return h.hexdigest()


def atomic_json(path,value,check=lambda:None):
    path=Path(path);tmp=path.with_name(path.name+'.tmp');check()
    try:
        with tmp.open('w') as f:
            for chunk in json.JSONEncoder(indent=2,allow_nan=False).iterencode(value):
                check();f.write(chunk)
            check();f.flush();os.fsync(f.fileno())
        check();os.replace(tmp,path)
    except BaseException:
        if tmp.exists():tmp.unlink()
        raise


def _group_exists(pgid):
    try:os.killpg(pgid,0);return True
    except ProcessLookupError:return False


def _reap_group(pgid):
    while True:
        try:
            pid,_=os.waitpid(-pgid,os.WNOHANG)
            if pid==0:return
        except ChildProcessError:return


def supervise(command,out,seconds,validator=None,env=None,started=None,preflight=None):
    """Only kills the new child session/process group; never global pkill.

    SIGKILL and machine failure are not catchable. Linux subreaper adopts
    same-group grandchildren for reaping; workers must not daemonize/setsid.
    """
    started=time.monotonic() if started is None else started
    deadline=started+seconds;out=Path(out);caught=[];child=None;old={};reason='failed';code=2
    status='failed';cleanup={};error=None
    def handler(s,f):caught.append(s)
    for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):old[s]=signal.signal(s,handler)
    subreaper=False
    if sys.platform.startswith('linux'):
        try:subreaper=ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)==0
        except (OSError,AttributeError):pass
    try:
        def check():
            if caught or (out/'STOP').exists():raise InterruptedError('controller stopping')
            if time.monotonic()>=deadline-6:raise TimeoutError('controller budget')
        if preflight:preflight(check)
        check()
        if caught:raise InterruptedError('stopped before spawn')
        child=subprocess.Popen(command,env=env,start_new_session=True)
        atomic_json(out/'controller_child.json',dict(pid=child.pid,pgid=child.pid),lambda:None)
        while True:
            if caught:reason=signal.Signals(caught[0]).name;code=128+caught[0];status='stopped';break
            if (out/'STOP').exists():reason='STOP';code=75;status='stopped';break
            if time.monotonic()>=deadline-6:reason='timeout';code=124;status='incomplete';break
            result=child.poll()
            if result is not None:
                code=result
                if result==0:
                    if validator:validator(check)
                    reason='completed';status='completed'
                else:reason='child_failed';status='failed'
                break
            time.sleep(.025)
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc);reason='controller_exception';status='failed';code=2
        if isinstance(exc,TimeoutError):reason='timeout';status='incomplete';code=124
        elif isinstance(exc,InterruptedError):reason='STOP';status='stopped';code=75
    finally:
        if child is not None:
            # Also clean descendants if their group leader has already exited.
            if _group_exists(child.pid):
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            grace=min(time.monotonic()+1.5,deadline-2)
            while _group_exists(child.pid) and time.monotonic()<grace:
                child.poll();_reap_group(child.pid);time.sleep(.01)
            if _group_exists(child.pid):
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            try:child.wait(timeout=max(.01,deadline-time.monotonic()-1))
            except subprocess.TimeoutExpired:
                reason='cleanup_timeout';status='incomplete';code=124
            until=min(time.monotonic()+.5,deadline-1)
            while time.monotonic()<until:
                _reap_group(child.pid)
                if not _group_exists(child.pid):break
                time.sleep(.01)
            cleanup=dict(direct_child_reaped=child.returncode is not None,owned_group_gone=not _group_exists(child.pid))
            if not all(cleanup.values()):status='incomplete';reason='cleanup_incomplete';code=124
        if caught:status='stopped';reason=signal.Signals(caught[0]).name;code=128+caught[0]
        elif (out/'STOP').exists() and status=='completed':status='stopped';reason='STOP';code=75
        if time.monotonic()>=deadline and status=='completed':status='incomplete';reason='deadline_before_commit';code=124
        record=dict(status=status,reason=reason,exit_code=code,caught_signals=[signal.Signals(s).name for s in caught],
                    cleanup=cleanup,subreaper_enabled=subreaper,controller_wall_seconds=time.monotonic()-started,
                    single_run_limit_seconds=seconds,no_absolute_old_deadline=True,SIGKILL_catchable=False,
                    training=0,scientific_chains=0,error=error)
        def final_check():
            if time.monotonic()>=deadline:raise TimeoutError('deadline during final controller save')
            if record['status']=='completed' and (caught or (out/'STOP').exists()):
                raise InterruptedError('termination during final controller save')
        try:
            try:atomic_json(out/'controller.json',record,final_check)
            except (InterruptedError,TimeoutError) as exc:
                # Emergency failure receipt only; no further scientific work.
                record.update(status='stopped' if isinstance(exc,InterruptedError) else 'incomplete',
                              reason='final_save_interrupted',exit_code=75 if isinstance(exc,InterruptedError) else 124)
                code=record['exit_code'];atomic_json(out/'controller.json',record,lambda:None)
        finally:
            for s,h in old.items():signal.signal(s,h)
    return code
