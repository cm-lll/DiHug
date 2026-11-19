import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from worker import *
import csv
def write(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    start=time.time();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    protocol=json.loads((OUT/'protocol.json').read_text());rows=[];relations=[];nodes=[];changes=[]
    for row in protocol['tasks']:
        folder=OUT/f"{row['dataset']}_{row['seed']}";source=Path(row['response_source']);old=OLD/folder.name
        assert json.loads((folder/'receipt.json').read_text())['forwards']==14
        train=torch.load(source/'update_input.pt',map_location='cpu',weights_only=False)
        fixed=torch.load(old/'evaluation_inputs.pt',map_location='cpu',weights_only=False)['inputs']
        data=np.load(row['data']);inp=fixed[0];ix=np.argsort(inp['ids']);core=HardScore(data['types'],inp['query'][:,ix],inp['f'][ix],inp['labels'][ix],len(data['schema']));exact=Exact(core)
        cached=list(csv.DictReader((source.parent/'evaluation_summary.csv').open()))
        for label,inp,j in [('same',train,-1)]+[(f't{x["t"]}',x,j) for j,x in enumerate(fixed)]:
            es=row['seed']+(protocol['same_input_evaluation_seed_offset'] if label=='same' else protocol['evaluation_hard_seed_offset']+j)
            uniform=np.random.default_rng(es).random((256,len(core.f)));baseline=np.load(source/f'{label}_before_scored.npz');B=baseline['values'];C=baseline['costs_remaining']
            for alpha in (0.,.25,.5,1.):
                if alpha in (0.,1.):
                    arm='before' if alpha==0 else 'after';s=next(x for x in cached if x['dataset']==row['dataset'] and int(x['seed'])==row['seed'] and x['panel']==label and x['arm']==arm)
                    z=np.load(source/f'{label}_{arm}_scored.npz');v=z['values'];cost=z['costs_remaining'];tri=float(s['expected_triangles'])
                    pz=np.load((source/'same_before.npz' if label=='same' else old/f'eval_before_t{inp["t"]}.npz') if alpha==0 else source/f'{label}_after.npz')
                else:
                    pz=np.load(folder/f'a{alpha}_{label}.npz');p=pz['probability'];V,_,stats=exact.value_gradient(p);v=np.zeros(5);v[[0,2,3]]=V;cost=remaining(core,(uniform<p).astype(np.uint8));v[[1,4]]=cost.mean(0);tri=sum(x['mean'] for x in stats['triangles'])
                    np.savez_compressed(folder/f'a{alpha}_{label}_scored.npz',values=v,costs_remaining=cost)
                p=pz['probability'];terms=pz['terms'];assert np.array_equal(pz['ids'],inp['ids'][np.argsort(inp['ids'])])
                expected=np.zeros((core.n,core.r));np.add.at(expected,(core.q[0],core.f),p);np.add.at(expected,(core.q[1],core.f),p);ref=core.ref['degree'];meta=dict(dataset=row['dataset'],seed=row['seed'],panel=label,t=inp['t'],alpha=alpha)
                pos=core.reference==1
                rows.append(dict(**meta,**{n:float(x) for n,x in zip(core.names,v)},mean_five=float(v.mean()),CE=float(terms.sum()*inp['ce_weight']),positive_CE=float(terms[pos].mean()),negative_CE=float(terms[~pos].mean()),expected_edges=float(p.sum()),expected_triangles=tri,node_total_degree_MAE=float(abs(expected.sum(1)-ref.sum(1)).mean())))
                for k,n in enumerate(core.names):changes.append(dict(**meta,metric=n,before=float(B[k]),value=float(v[k]),change=float(v[k]-B[k]),analytic=k in (0,2,3),paired_MC_SE=float((cost-C)[:,[1,4].index(k)].std(ddof=1)/16) if k in (1,4) else 0.))
                for f in range(core.r):
                    m=core.f==f;a=m&pos;b=m&~pos;relations.append(dict(**meta,family=f,positive_n=int(a.sum()),negative_n=int(b.sum()),reference_edges=int(core.ref['counts'][f]),expected_edges=float(p[m].sum()),missed_mass=float((1-p[a]).sum()),false_mass=float(p[b].sum()),positive_CE=float(terms[a].mean()) if a.any() else None,negative_CE=float(terms[b].mean()) if b.any() else None))
                d=ref.sum(1)
                for ty in np.unique(core.types):
                    m=core.types==ty;cut=float(np.quantile(d[m],.9));hub=m&(d>=cut);low=m&(d<=2)&~hub
                    for group,sel in [('low',low),('middle',m&~hub&~low),('hub',hub)]:
                        nodes.append(dict(**meta,node_type=int(ty),group=group,n=int(sel.sum()),reference_hub_threshold=cut,total_degree_MAE=float(abs(expected.sum(1)-d)[sel].mean()) if sel.any() else None,relation_degree_MAE=float(abs(expected-ref)[sel].mean()) if sel.any() else None))
    for p,h in json.loads((OUT/'bindings.json').read_text()).items():assert rt.sha(p)==h,p
    write('evaluation_summary.csv',rows);write('risk_changes.csv',changes);write('relations.csv',relations);write('node_groups.csv',nodes)
    rt.save(OUT/'audit.json',dict(passed=True,rows=len(rows),new_forwards=56,updates=0,backwards=0,CPU_seconds=time.time()-start,bound_assets_unchanged=True,endpoint_predictions_reused=True))
if __name__=='__main__':main()
