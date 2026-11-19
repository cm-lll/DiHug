"""Rehash prepared assets and freeze actual runtime sources; no model calls."""
import json,os,hashlib,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=ROOT.parents[1];OUT=Path(os.environ['STRUCTURE_RUN'])
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    expected=read(OUT/'asset_manifest_expected.json');rows=[]
    for r in expected['tasks']:
        for path,key in [('initial','initial_sha256'),('readout','readout_sha256'),('training_log','training_log_sha256')]:
            assert sha(r[path])==r[key],r[path];rows.append(r[path])
    for r in expected['graphs']:assert sha(r['file'])==r['sha256'];rows.append(r['file'])
    pb=P/'results/pb_degree_formal_20261106T170600'
    for t in read(OUT/'tasks.json'):
        d=pb/'run'/t['task_id'];ro=d/'readout_ema_0.99.pt'
        assert sha(ro)==read(ro.with_suffix('.sha256.json'))['sha256'];rows.append(str(ro))
        for ss in [2026102821,2026102822,2026110231,2026110232]:
            f=d/'ema_0.99'/f'sample_{ss}';b=read(f/'binding.json')
            assert b['dataset']==t['dataset'] and b['train_seed']==t['train_seed'] and b['sample_seed']==ss and b['updates']==3000 and b['configuration']=='A_PB_late' and b['readout']=='EMA0.99'
            assert sha(f/'graph.npz')==b['graph_sha256'] and sha(ro)==b['model_sha256'];rows.append(str(f/'graph.npz'))
            for version in ['original','lcc']:
                cached=read(pb/'evaluation'/(t['task_id']+'_PB')/'u3000'/str(ss)/(version+'.json'))
                assert cached['graph_binding']['graph_sha256']==b['graph_sha256']
    sources=read(P/'results/cn_multiseed_20261104T042546/source_manifest.json')
    for p,h in sources.items():assert sha(p)==h,p
    # Include all local executable modules and baseline modules actually importable.
    source_root=P/'research_branches/cn_multiseed_20261104'
    for tree in [source_root/'experiments/o_cn_matched_execute_v1',source_root/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1']:
        for p in tree.rglob('*.py'):sources[str(p)]=sha(p)
    for p in ROOT.rglob('*.py'):sources[str(p)]=sha(p)
    (OUT/'source_binding.json').write_text(json.dumps(sources,indent=2)+'\n')
    (OUT/'asset_audit.json').write_text(json.dumps(dict(passed=True,files_rehashed=rows,source_files=len(sources),disk_free_GiB=shutil.disk_usage(P).free/2**30,epoch=time.time()),indent=2)+'\n')
    print('intake passed',len(rows),len(sources))
if __name__=='__main__':main()
