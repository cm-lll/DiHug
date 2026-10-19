"""Device migration is explicit; numerical dependencies/source are NEVER relaxed."""
import json
from pathlib import Path
import time
from .data import digest

def scientific(binding):
    result={k:v for k,v in binding.items() if k!='device_identity'}
    result['runtime']={k:v for k,v in binding['runtime'].items() if k!='device_name'}
    return result

def check(origin,current,allow=False):
    if scientific(origin)!=scientific(current):
        raise ValueError('source/config/data/runtime software mismatch; device migration cannot bypass it')
    changed=origin!=current
    if changed and not allow:raise ValueError('device identity mismatch; explicit --allow-device-migration required')
    return changed

def record(out,source,current,source_sha,scope,write,rng_restored=True):
    out=Path(out)/'migrations';out.mkdir(exist_ok=True)
    body=dict(source_execution_binding=source,target_execution_binding=current,
        source_sha256=source_sha,scope=scope,source_untouched=True,
        numerical_dependencies_verified=True,rng_restored=rng_restored,
        rng_restore='saved selected-device state restored to current selected device' if rng_restored else 'model-only import: training RNG not restored',
        bitwise_cross_device_reproducibility=False)
    path=out/(digest(body)+'.json')
    if path.exists():
        existing=json.loads(path.read_text())
        if {k:v for k,v in existing.items() if k!='recorded_utc'}!=body:raise ValueError('migration receipt conflict')
    else:write(path,dict(body,recorded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    return path
