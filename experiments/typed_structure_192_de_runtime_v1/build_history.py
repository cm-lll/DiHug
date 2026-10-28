"""Read-only extraction of already published result payloads; no re-evaluation."""
import argparse
import hashlib
import json
import pathlib
import zipfile

SOURCES = [
    ('round1', 'b5b697361e389cd77d2fb09a84e77c0b6915407d', 'a200e9246c57a0bef873bf8ed71bc3e4e5450b2432dfa94454a12f7734ada0a1', 'run', {'IMDB':2026102811,'PubMed':2026102812}),
    ('round2', '143e6bf80a9858ec21e512da2f1a03e02e3a5a38', '6e4b4249ab012972a7a2e47ddd7e6d7513837ec47f71facd9ba375da46462c4e', 'round2/run', {'IMDB':2026102834,'PubMed':2026102833}),
]

def digest(data):
    return hashlib.sha256(data).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--round1',required=True);p.add_argument('--round2',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    output=pathlib.Path(a.output);output.mkdir(parents=True,exist_ok=False)
    index=dict(sources=[],records=[],files={},checkpoints='historical hash receipts only; private weight bytes not re-certified here')
    for label,commit,expected,prefix,seeds in SOURCES:
        archive=pathlib.Path(getattr(a,label));assert digest(archive.read_bytes())==expected,'published archive hash mismatch'
        with zipfile.ZipFile(archive) as z:
            root=z.namelist()[0].split('/')[0]
            manifest_bytes=z.read(root+'/PAYLOAD_MANIFEST.json');manifest={r['path']:r for r in json.loads(manifest_bytes)['files']}
            index['sources'].append(dict(round=label,commit=commit,zip_sha256=expected,manifest_sha256=digest(manifest_bytes)))
            def take(relative):
                data=z.read(root+'/'+relative);m=manifest[relative]
                assert digest(data)==m['sha256'] and len(data)==m['bytes'],relative
                target=pathlib.Path(label)/relative[len(prefix)+1:]
                dest=output/target;dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():dest.write_bytes(data)
                elif dest.read_bytes()!=data:raise ValueError('conflicting historical payload')
                index['files'][target.as_posix()]=dict(sha256=digest(data),bytes=len(data),archive_member=root+'/'+relative,source_commit=commit,archive_sha256=expected)
                return target.as_posix(),data
            for dataset,seed in seeds.items():
                for arm in 'OABC':
                    cell=prefix+'/'+dataset+'_'+arm
                    cell_files={}
                    for name in ['binding.json','summary.json','final.sha256.json','initialization.json','training_rng_reset.json']:
                        path,data=take(cell+'/'+name);cell_files[name]=path
                    binding=json.loads((output/cell_files['binding.json']).read_text())
                    assert binding['arm']==arm and binding['config']['seed']==seed
                    for sample in [2026102821,2026102822]:
                        paths={name:take(cell+'/sample_'+str(sample)+'/'+name)[0] for name in ['graph.npz','metrics.json','binding.json']}
                        b=json.loads((output/paths['binding.json']).read_text());final=json.loads((output/cell_files['final.sha256.json']).read_text())
                        assert b['seed']==sample and b['binding']==binding and b['checkpoint_sha256']==final['sha256']
                        assert b['graph_sha256']==index['files'][paths['graph.npz']]['sha256']
                        index['records'].append(dict(dataset=dataset,train_seed=seed,sample_seed=sample,arm=arm,round=label,source_commit=commit,paths=paths,cell_files=cell_files))
    (output/'index.json').write_text(json.dumps(index,indent=2,sort_keys=True))
    print(json.dumps(dict(records=len(index['records']),files=len(index['files']),index_sha256=digest((output/'index.json').read_bytes()))))

if __name__=='__main__':main()
