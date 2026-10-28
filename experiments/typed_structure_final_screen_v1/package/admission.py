"""Shared immutable approval inputs and deterministic slot scheduling."""
import pathlib,json,hashlib
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dictionaries(directory,approval):
    for dataset in ['IMDB','PubMed']:
        root=pathlib.Path(directory)/('dictionary_'+dataset);e=approval['dictionaries'][dataset]
        for name in ['final_basis.npz','selection.json','panel.json','frozen_splits.json']:
            if sha(root/name)!=e['files'][name]:raise ValueError('external approval hash differs '+dataset+'/'+name)
        r=json.loads((root/'selection.json').read_text())
        if r['status']!='passed' or r['K']!=e['K'] or r['offsets']!=e['offsets'] or r['offsets'][-1]!=e['dimension']:raise ValueError('dictionary approval dimensions/status')
from runtime_plan import assignment

def verify_summary(package_root,approval):
    path=pathlib.Path(package_root).parent/'summarize.py'
    expected=approval.get('summarize_sha256')
    if not expected or sha(path)!=expected:raise ValueError('approved external summarize.py hash mismatch')
    return path
