"""No protected legacy workers in this new eight-card queue."""
import pathlib
def filter_passive(handles,devices):
    unresolved=[];exited=[]
    for pid in sorted(set(handles.split())):
        if not pid.isdecimal():raise ValueError('unexpected fuser output')
        if (pathlib.Path('/proc')/pid).exists():unresolved.append(pid)
        else:exited.append(dict(pid=int(pid),reason='exited_before_inspection'))
    return ' '.join(unresolved),exited
