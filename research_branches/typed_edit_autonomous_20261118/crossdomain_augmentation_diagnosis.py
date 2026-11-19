"""Read saved banks/endpoints to freeze one cross-relation coverage hypothesis."""
import confirmation_fit as fit
import additional_confirmation_fit as additional
import confirmation_common as contract
from candidate_consequences import Consequences
from round01 import b,np,torch,time,Path,ROOT,SRC

OUT=ROOT/'results/typed_edit_crossdomain_augmentation_20261118/diagnosis'
BASE=ROOT/'results/typed_edit_autonomous_20261118'

def assets(ds,repeat):
    if ds=='pubmed':
        cache=fit.CACHE/ds/f'repeat{repeat}';old=fit.OUT/f'inputs/{ds}_old_train.npz'
        cp=fit.OUT/f'fits/{ds}/{repeat}/checkpoint.pt';schema=fit.CACHE/ds/'schema.json'
        deploy=BASE/'confirmation_v1/deploy';prefix=BASE/'confirmation_v1/prefix_v2'
        def folder(seed,arm):return deploy/f'outputs/{ds}/{seed}_C/{arm}_{repeat if arm in ["F","Q"] else 0}'
        seeds=[12001,12002]
    else:
        cache=BASE/f'additional_confirmation_v1/cache/{ds}/{repeat}';old=BASE/f'additional_old_cache/{ds}/old_train.npz'
        cp=BASE/f'additional_confirmation_v1/fit/{ds}/{repeat}/checkpoint.pt';schema=cache/'schema.json'
        deploy=BASE/f'additional_confirmation_v1/deploy/{ds}';prefix=BASE/f'additional_confirmation_v1/prefix/{ds}'
        def folder(seed,arm):return deploy/f'outputs/{seed}_C/{arm}_{repeat if arm in ["F","Q"] else 0}'
        seeds=[15001,15002]
    return dict(cache=cache,old=old,checkpoint=cp,schema=schema,deploy=deploy,prefix=prefix,folder=folder,seeds=seeds)

def main():
    b.setup();OUT.mkdir(parents=True,exist_ok=False);tic=time.monotonic();files=[Path(__file__),SRC/'CROSSDOMAIN_AUGMENTATION_VALIDATION.md']
    specs=[]
    for ds in ['pubmed','acm']:
        repeats=[1,2,3] if ds=='pubmed' else [2,3,4]
        for repeat in repeats:
            a=assets(ds,repeat);specs.append((ds,repeat,a))
            files += [a['cache']/'new_train.npz',a['cache']/'normalization.json',a['old'],a['checkpoint'],a['schema'],a['deploy']/'acceptance.json',a['deploy']/'manifest.json',b.RUN/f'data/{ds}/graph.json']
            for seed in a['seeds']:
                for arm in ['F','Q']:files.append(a['folder'](seed,arm)/'graph2048.json')
    contract.bind(OUT,files,new_fits=0,new_graphs=0,selection_uses_existing_outputs=True)
    coverage=[];detail=[];selected={}
    for ds in ['pubmed','acm']:
        ss=[s for s in specs if s[0]==ds];cols=b.read(ss[0][2]['schema'])['columns'];ref=b.graph(ds);refraw=Consequences(ref,cols).counts
        allcounts={arm:[] for arm in ['F','Q']}
        for _,repeat,a in ss:
            fit.source_check(a['deploy'])
            cp=torch.load(a['checkpoint'],map_location='cpu',weights_only=False)['state_dict'];k=len(cols)
            for kind,path in [('old',a['old']),('new',a['cache']/'new_train.npz')]:
                z=np.load(path);raw=z['counts'] if 'counts' in z.files else z['x'][:,0,:k]
                delta=z['delta'][np.arange(len(z['y'])),z['y']] if 'delta' in z.files else z['x'][np.arange(len(z['y'])),z['y'],k:]
                for j,(family,key) in enumerate(cols):
                    coverage.append(dict(dataset=ds,repeat=repeat,bank=kind,family=family,key=key,records=len(raw),reference=int(refraw[j]),minimum=int(raw[:,j].min()),maximum=int(raw[:,j].max()),above_negative=int(((raw[:,j]>refraw[j])&(delta[:,j]<0)).sum()),below_positive=int(((raw[:,j]<refraw[j])&(delta[:,j]>0)).sum())))
            for seed in a['seeds']:
                for arm in ['F','Q']:
                    graph=b.restore_graph(b.read(a['folder'](seed,arm)/'graph2048.json'));raw=Consequences(graph,cols).counts;allcounts[arm].append(raw)
            for j,(family,key) in enumerate(cols):
                aa=float(cp['weights'][j]);bb=float(cp['weights'][k+j]);mu=float(cp['mu'][j]);sd=float(cp['sd'][j])
                detail.append(dict(dataset=ds,repeat=repeat,family=family,key=key,linear=aa,quadratic=bb,mu=mu,sd=sd,stationary_count=mu-aa*sd/(2*bb) if bb>1e-12 else None))
        ff=np.stack(allcounts['F']);qq=np.stack(allcounts['Q']);ranks=[]
        for j,(family,key) in enumerate(cols):
            denom=max(1,sum(refraw[i] for i,col in enumerate(cols) if col[0]==family))
            gap=float((abs(ff[:,j]-refraw[j])-abs(qq[:,j]-refraw[j])).mean()/denom)
            row=dict(family=family,key=key,index=j,reference=int(refraw[j]),F_mean=float(ff[:,j].mean()),Q_mean=float(qq[:,j].mean()),absolute_error_gap=gap)
            if family=='cross_relation_four_cycles':ranks.append(row)
        ranks.sort(key=lambda r:(-r['absolute_error_gap'],r['key']));top=ranks[0]
        direction=1 if top['F_mean']>=top['reference'] else -1
        counts=[r['above_negative'] if direction==1 else r['below_positive'] for r in coverage if r['dataset']==ds and r['bank']=='new' and r['key']==top['key'] and r['family']==top['family']]
        selected[ds]=dict(**top,forward_direction=direction,needed_inverse_direction=-direction,new_bank_corrective_coverage=counts,ranking=ranks,rule='largest positive F-minus-Q absolute error gap among relation four-cycle coordinates; ties lexicographic; forward direction follows F bias',development_starts=ss[0][2]['seeds'],positive_gap=top['absolute_error_gap']>0)
    contract.verify(OUT);contract.put(OUT,'coverage.json',coverage);contract.put(OUT,'coefficients.json',detail);contract.put(OUT,'summary.json',dict(selected=selected,new_fits=0,new_graphs=0,causal_explanation_not_proven=True))
    contract.put(OUT,'acceptance.json',dict(passed=True,seconds=time.monotonic()-tic,outputs={str(OUT/n):b.sha(OUT/n) for n in ['summary.json','coverage.json','coefficients.json']}))
    print(selected,flush=True)

if __name__=='__main__':main()
