"""Read-only cache acceptance; fixes list subtraction in the independent checker."""
import argparse
import crossdomain_augmentation_cache as original
import confirmation_common as contract
from round01 import b,np,time,Path,ROOT,SRC
BASE=original.BASE

def audit(ds,z,report,cols,deadline):
    ref=b.graph(ds);edges,nb=original.original.c.r5.topology(ref);names=sorted(edges);selected=0;rec={r['step']:r for r in report['records']};r5=original.original.c.r5
    for entry in report['path']:
        assert time.monotonic()<deadline
        step=entry['step'];before={f:set(es) for f,es in edges.items()} if step in rec else None
        r5.edit(edges,nb,names,entry['edit'],0)
        if step not in rec:continue
        r=rec[step];i=selected;g=r5.graph_from(ref,edges);_,actual=original.original.c.context(g,cols)
        actual_counts=np.asarray(actual['raw'],dtype=np.int64)
        assert np.array_equal(actual_counts,z['counts'][i]) and np.array_equal(actual_counts,r['counts'])
        assert int(z['y'][i])==r['label']
        assert np.array_equal(original.encode(actual_counts,z['delta'][i],ref.num_nodes,cols),z['x'][i])
        for f in names:assert np.array_equal(g.family_degrees(f),ref.family_degrees(f))
        vals=r['candidates'];label=r['label'];dd=z['delta'][i]
        tocheck=range(16) if step%32==0 else [label]
        for j in tocheck:
            r5.edit(edges,nb,names,vals,j);_,new=original.original.c.context(r5.graph_from(ref,edges),cols)
            assert np.array_equal(np.asarray(new['raw'],dtype=np.int64)-actual_counts,dd[j]);r5.edit(edges,nb,names,vals,j,True)
        r5.edit(edges,nb,names,vals,label);assert edges==before;r5.edit(edges,nb,names,vals,label,True)
        for f in names:assert edges[f]=={tuple(sorted(e)) for e in g.family_edges[f]}
        selected+=1
    final=b.restore_graph(report['final_graph'])
    for f in names:assert edges[f]=={tuple(sorted(e)) for e in final.family_edges[f]}
    assert selected==len(z['y']);return dict(passed=True,steps=len(report['path']),records=selected)

def verify_producer(parent):
    # A new consumer verifies the producer's frozen files, not its own module set.
    for p,h in b.read(parent/'manifest.json')['files'].items():assert b.sha(p)==h,p

def main(ds,engineering_only=False):
    b.setup();a=original.diagnosis.assets(ds,1 if ds=='pubmed' else 2);cols=b.read(a['schema'])['columns']
    if engineering_only:
        parent=BASE/f'engineering/{ds}/bank999';checks=[]
        for kind in ['targeted','ordinary']:
            checks.append(dict(kind=kind,**audit(ds,np.load(parent/f'{kind}.npz'),b.read(parent/f'{kind}.json'),cols,time.monotonic()+180)))
        contract.put(BASE,f'engineering_replay_{ds}.json',dict(passed=True,checks=checks,new_fits=0,new_graphs=0,checker_source_sha256=b.sha(Path(__file__))));print(ds,'engineering replay passed',flush=True);return
    parent=BASE/f'cache/{ds}/bank1';assert (parent/'completion.json').exists() and not (parent/'acceptance.json').exists()
    failure=b.read(parent/'failure.json');assert failure['type']=='TypeError' and 'unsupported operand' in failure['message']
    control=BASE/f'followthrough/{ds}/completion.json';assert control.exists() and b.read(control)['phases']['cache'] is False
    verify_producer(parent);out=BASE/f'cache_finalized/{ds}/bank1';out.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),parent/'manifest.json',parent/'completion.json',parent/'failure.json',control]
    files+=list(parent.glob('*/trajectories/*/record.json'))+list(parent.glob('*/trajectories/*/data.npz'))+list(parent.glob('*/augmentation.npz'))
    contract.bind(out,files,new_graphs=0,new_fits=0,fix='np.asarray for JSON count vectors in independent replay; scientific source and outputs unchanged')
    tic=time.monotonic();checks=[];rows=[]
    for kind in ['targeted','ordinary']:
        blocks=[];records=[]
        for tid in range(8):
            folder=parent/f'{kind}/trajectories/{tid}';z=np.load(folder/'data.npz');r=b.read(folder/'record.json');check=audit(ds,z,r,cols,tic+1800)
            checks.append(dict(kind=kind,trajectory=tid,**check));contract.put(out,f'checks/{kind}_{tid}.json',checks[-1]);blocks.append(z);records.extend(r['records'])
            print(ds,'read-only accepted',kind,tid,flush=True)
        merged=np.load(parent/f'{kind}/augmentation.npz')
        for key in blocks[0].files:assert np.array_equal(merged[key],np.concatenate([z[key] for z in blocks]))
        rows.append(dict(kind=kind,records=len(merged['y']),covered=sum(r['covered'] for r in records)))
    passed=rows[0]['covered']>=128;contract.verify(out);verify_producer(parent)
    summary=dict(passed=passed,rows=rows,selection=b.read(original.diagnosis.OUT/'summary.json')['selected'][ds],normalization_changed=False,read_only_finalizer=str(out))
    acceptance=dict(passed=True,coverage_gate_passed=passed,checks=checks,seconds=time.monotonic()-tic,read_only_finalizer=str(out),new_fits=0,new_graphs=0,outputs={str(parent/f'{k}/augmentation.npz'):b.sha(parent/f'{k}/augmentation.npz') for k in ['targeted','ordinary']})
    contract.put(out,'summary.json',summary);contract.put(out,'acceptance.json',acceptance)
    contract.put(parent,'summary.json',summary);contract.put(parent,'acceptance.json',acceptance)
    print(ds,'cache read-only finalized',rows,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['pubmed','acm']);p.add_argument('--engineering-only',action='store_true');a=p.parse_args();main(a.dataset,a.engineering_only)

