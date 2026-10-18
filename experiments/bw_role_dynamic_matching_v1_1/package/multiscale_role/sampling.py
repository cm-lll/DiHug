"""Same frozen-step integration, extended binding for the new objective code."""
import hashlib
from pathlib import Path
from stepwise_role.sampling import reverse_step,save_checkpoint,load_checkpoint,validate_arm
from stepwise_role.sampling import binding as old_binding
from stepwise_role.core import signature


def binding(*args,**kwargs):
    value=old_binding(*args,**kwargs)
    folder=Path(__file__).resolve().parent
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob('*.py'))}
    value['multiscale_code_sha256']=signature(files)
    return value
