"""Read-only admission checks, before any output directory/file is created."""
import json
from pathlib import Path
from multiscale_role.runtime_guard import checked_sha

SEALED_LEDGER_SHA256='961e6d93c8194396223ac9062b1658ac87232e82710b1ab3d36e722ed2c00196'


def overlap(left,right):
    left,right=Path(left).resolve(),Path(right).resolve()
    return left==right or left in right.parents or right in left.parents


def validate_output_path(action,out,root,*,legacy_run=None,legacy_zip=None,manifest=None,evaluation=None,check=lambda:None):
    """Resolve symlink aliases and reject overlap in either direction.

    A source ZIP is a protected file, not its whole containing directory:
    sibling outputs beside an uploaded ZIP remain permitted. Original results,
    state/checkpoint roots and converted input directories are protected trees.
    Metadata is only read here, never repaired or written.
    """
    check();out=Path(out).resolve();trees=[Path(root).resolve()];files=[]
    def trajectory_roots(activation):
        if activation.is_file():
            check();data=json.loads(activation.read_text());check()
            for key in ('states','checkpoints'):
                if key in data:trees.append(Path(data[key]).resolve())
    if action=='export-legacy':
        old=Path(legacy_run).resolve();trees.append(old)
        if legacy_zip:files.append(Path(legacy_zip).resolve())
        # Reject the known tree FIRST, even when its metadata is missing/bad.
        if overlap(out,old):raise ValueError('output overlaps protected input tree: '+str(old))
        trajectory_roots(old/'activation.json')
    elif action=='short':
        for arg in (manifest,evaluation):
            if arg:trees.append(Path(arg).resolve().parent);files.append(Path(arg).resolve())
        if manifest and Path(manifest).is_file():
            parent=Path(manifest).resolve().parent
            data=json.loads(Path(manifest).read_text())
            for row in data.get('records',[]):
                check()
                for key in ('checkpoint','source_manifest','config','npz','metadata'):
                    if key in row:
                        p=Path(row[key]);p=(p if p.is_absolute() else parent/p).resolve();files.append(p)
                        if key=='checkpoint':trees.append(p.parent)
                meta_path=Path(row.get('metadata',''))
                meta_path=(meta_path if meta_path.is_absolute() else parent/meta_path)
                if meta_path.is_file():
                    meta=json.loads(meta_path.read_text())
                    for path in meta.get('original_input_hashes',{}):
                        p=Path(path).resolve();files.append(p)
                        if p.name=='evidence_hashes.json':
                            trees.append(p.parent);trajectory_roots(p.parent/'activation.json')
    for p in trees+files:
        check()
        if overlap(out,p):raise ValueError('output overlaps protected input: '+str(p))
    return dict(output=str(out),protected_trees=sorted(set(map(str,trees))),protected_files=sorted(set(map(str,files))))


def verify_sealed_ledger(root,expected,check=lambda:None):
    if expected!=SEALED_LEDGER_SHA256:raise ValueError('ledger expected hash differs from user-supplied sealed value')
    path=Path(root)/'evidence_hashes.json'
    actual=checked_sha(path,check)
    if actual!=expected:raise ValueError('sealed ledger mismatch; original ledger must not be rewritten')
    return actual


def test_environment(environ):
    # Prevent inherited -k/-m/--ignore/--deselect, plugin or addopts injection.
    result={k:v for k,v in environ.items() if not k.startswith('PYTEST_')}
    result['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
    result['PYTHONDONTWRITEBYTECODE']='1'
    return result
