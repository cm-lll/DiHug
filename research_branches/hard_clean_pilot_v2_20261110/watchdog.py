"""Independent resource hard stop. Only registered own process groups are touched."""
import os,sys,time,json,signal
from pathlib import Path
out=Path(sys.argv[1]);window=json.loads((out/'window.json').read_text());stop_at=None
while not (out/'DONE').exists():
    p=out/'allocations.json'
    allocations=json.loads(p.read_text()) if p.exists() else []
    now=time.time();spent=sum((a['end_epoch'] or now)-a['start_epoch'] for a in allocations)
    if now>=window['science_stop_epoch']-30 or spent>=window['GPU_seconds_limit']-30:
        (out/'STOP').touch()
        if stop_at is None:stop_at=now
    if (out/'STOP').exists():
        if stop_at is None:stop_at=now
        for a in allocations:
            if a['end_epoch'] is not None:continue
            proc=Path(f'/proc/{a["pid"]}/stat')
            try:
                if proc.read_text().split()[21]!=a['start_ticks']:continue
                os.killpg(a['pid'],signal.SIGKILL if now-stop_at>20 else signal.SIGTERM)
            except (FileNotFoundError,ProcessLookupError):pass
    time.sleep(2)

