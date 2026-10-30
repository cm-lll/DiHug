"""Standard-library-only bundle integrity checker. Never imports model modules."""
import ast, hashlib, json, pathlib

def main():
    root=pathlib.Path(__file__).resolve().parent
    manifest=json.loads((root/'MANIFEST.json').read_text())
    checked=0;syntax=0
    for name,row in manifest.items():
        path=root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root):raise ValueError('unsafe path: '+name)
        data=path.read_bytes()
        if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('payload mismatch: '+name)
        if path.suffix=='.py':ast.parse(data,filename=name);syntax+=1
        checked+=1
    forbidden=[str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.suffix in ('.pt','.pth','.ckpt','.pem','.key')]
    if forbidden:raise ValueError('unexpected weights/credentials: '+repr(forbidden))
    bound=0
    for snapshot in ('ac_screen','early_count_stage500','early_count_length'):
        original=json.loads((root/'provenance'/f'{snapshot}_actual_execution_manifest.json').read_text())
        for name,digest in original['files'].items():
            if hashlib.sha256((root/'snapshots'/snapshot/name).read_bytes()).hexdigest()!=digest:raise ValueError('original execution mismatch: '+snapshot+'/'+name)
            bound+=1
    print(json.dumps(dict(status='passed',payload_files=checked,python_syntax_files=syntax,actual_execution_hash_matches=bound,models_loaded=0,gpu_calls=0,training_updates=0,graphs_generated=0)))

if __name__=='__main__':main()
