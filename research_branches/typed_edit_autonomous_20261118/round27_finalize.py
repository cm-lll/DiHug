"""Read-only scientific replay; repair heterogeneous CSV columns in a new output."""
import round27_deploy as prev
from round01 import b,np,torch,time,Path,ROOT,SRC,json,csv,t
from potential import encode
from candidate_consequences import Consequences
OUT=ROOT/'results/typed_edit_autonomous_20261118/r27_finalize'

def put(name,v):
    f=OUT/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def verify():
    prev.verify()
    for f,h in b.read(OUT/'manifest.json')['files'].items():assert b.sha(f)==h,f

def main():
    b.setup();prev.verify();assert b.read(prev.OUT/'completion.json')['completed'];OUT.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),SRC/'ROUND27_FINALIZE.md',prev.OUT/'summary.json',prev.OUT/'failure.json',prev.OUT/'completion.json']
    for job in prev.JOBS:files.extend((prev.OUT/'outputs'/job['dataset']/str(job['start_id'])).glob('*'))
    put('manifest.json',dict(files={str(f):b.sha(f) for f in files if f.is_file()},new_graphs=0,new_training_updates=0));tic=time.monotonic();error=0.;full=0;replayed=0
    for job in prev.JOBS:
        ds=job['dataset'];sid=job['start_id'];folder=prev.OUT/'outputs'/ds/str(sid);tr=b.read(folder/'trace.json');z=np.load(folder/'features.npz');g,target,_=prev.inputs(job);edges,nb=prev.r5.topology(g);names=sorted(edges);_,_,cols,_=prev.p.spec(ds);m=prev.model(ds)
        with torch.no_grad():
            xx=np.stack([encode(raw,dd,g.num_nodes,cols) for raw,dd in zip(z['counts'],z['delta'])]);q=m(torch.from_numpy(xx).double()).numpy()
        error=max(error,float(abs(q-z['logits']).max()));assert error<=1e-9
        for i,v in enumerate(tr):
            assert time.monotonic()-tic<900
            vals=v['candidates'];raw=z['counts'][i];dd=z['delta'][i]
            if i%32==0:
                graph=prev.r5.graph_from(g,edges);_,fresh=prev.cache.context(graph,cols);assert np.array_equal(fresh['raw'],raw);assert np.array_equal(Consequences(graph,cols,fresh['raw']).differences(t.proposals(graph,vals)),dd);full+=1
            chosen=v['chosen'];assert chosen==int(q[i].argmax());prev.r5.edit(edges,nb,names,vals,chosen);reject=not prev.r5.connected(nb);assert reject==v['connectivity_rejected']
            if reject:prev.r5.edit(edges,nb,names,vals,chosen,True)
            effective=0 if reject else chosen;assert effective==v['effective'] and prev.r5.connected(nb)
            if i+1<len(tr):assert np.array_equal(raw+dd[effective],z['counts'][i+1])
            if i+1 in [128,512]:
                graph=prev.r5.graph_from(g,edges);stored=b.restore_graph(b.read(folder/f'graph{i+1}.json'))
                for f in names:assert edges[f]=={tuple(sorted(e)) for e in stored.family_edges[f]} and np.array_equal(graph.family_degrees(f),g.family_degrees(f))
                report=b.read(folder/f'report{i+1}.json')
                for key,value in prev.metrics(ds,graph,target).items():assert report[key]==value
            replayed+=1
        print('R27 replayed',job,flush=True)
    summary=b.read(prev.OUT/'summary.json');rows=summary['rows'];fields=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/'per_output.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    put('summary.json',summary);put('completion.json',dict(completed=True,new_graphs=0,new_training_updates=0,reused_scientific_graphs=4,scientific_opportunities=0))
    verify();put('acceptance.json',dict(passed=True,replayed_opportunities=replayed,sampled_full_state_recounts=full,snapshot_metric_recomputations=8,logit_error=error,seconds=time.monotonic()-tic));print('R27 accepted',summary['decisions'],flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:put('failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
