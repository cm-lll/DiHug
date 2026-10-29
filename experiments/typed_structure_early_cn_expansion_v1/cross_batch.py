"""Read-only union, identity-aware comparison. Never loads a model or graph."""
import json,math,pathlib,argparse,copy
from runtime_contract import ROOT,PROTOCOL,read,sha,source_binding

def mean(x):return sum(x)/len(x) if x and all(v is not None and math.isfinite(float(v)) for v in x) else None
def difference(a,b):return a-b if a is not None and b is not None else None
def key(r):return (r['dataset'],r['train_seed'],r['arm'],r['sample_seed'],r['version'])
def cells():
    out=[]
    for d in PROTOCOL['train_seeds']:
        old=PROTOCOL['train_seeds'][d];new=PROTOCOL['new_train_seeds'][d]
        out.extend((d,old,a) for a in ('D_earlyCN','E_earlyCN','D_lateCN','E_lateCN'))
        out.extend((d,new,a) for a in ('E_earlyCN','E_lateCN'))
        if d=='PubMed':out.append((d,old,'C'))
    return out
def comparisons():
    out=[]
    for d,old in PROTOCOL['train_seeds'].items():
        for a,b in [('E_earlyCN','E_lateCN'),('D_earlyCN','D_lateCN'),('E_earlyCN','D_earlyCN'),('E_lateCN','D_lateCN')]:
            out.append((d,old,a,b))
        out.append((d,PROTOCOL['new_train_seeds'][d],'E_earlyCN','E_lateCN'))
        if d=='PubMed':
            out.extend((d,old,a,'C') for a in ('E_earlyCN','D_earlyCN'))
    return out
def normalize(r,arm,batch):
    value=copy.deepcopy(r);value.update(arm=arm,batch=batch,model=arm.split('_')[0],fusion='early' if 'early' in arm else 'late' if 'late' in arm else 'none')
    return value
def external_records():
    result=[];inventory=[]
    b=read(ROOT/'baseline_bindings.json')['old_evaluation']
    if sha(b['path'])!=b['sha256']:raise ValueError('old frozen evaluation bytes')
    for r in read(b['path'])['records']:
        d=r['dataset']
        allowed=('D','E','C') if d=='PubMed' else ('D','E')
        if r['train_seed']==PROTOCOL['train_seeds'][d] and r['arm'] in allowed:
            result.append(normalize(r,r['arm']+'_lateCN' if r['arm'] in ('D','E') else 'C','old_final_screen'))
    if len(result)!=20:raise ValueError('old controls missing/duplicated')
    inventory.append(dict(batch='old_final_screen',status='sealed',records=len(result),sha256=b['sha256']))
    a=read(ROOT/'cross_batch_binding.json');run=pathlib.Path(a['run'])
    if sha(run/'activation.json')!=a['activation_sha256'] or sha(a['approval_path'])!=a['approval_sha256']:raise ValueError('protected original run binding changed')
    q=run/'unified_evaluation.json';receipt=run/'launcher_summary.json'
    if not q.exists() or not receipt.exists() or read(receipt).get('status')!='complete':
        inventory.append(dict(batch='protected_two_E_early',status='not_sealed_or_incomplete',reason='running/failed/unpublished; do not import live partial metrics',summary_exists=q.exists(),launcher_status=read(receipt).get('status') if receipt.exists() else None))
        return result,inventory
    value=read(q)
    if value.get('source_sha256')!=a['source_sha256'] or value.get('status')!='complete':raise ValueError('protected evaluation identity')
    selected=[normalize(r,'E_earlyCN','protected_two_E_early') for r in value['records'] if r['arm']=='E_earlyCN' and r['train_seed']==PROTOCOL['train_seeds'][r['dataset']]]
    expected={(d,s,'E_earlyCN',sample,v) for d,s in PROTOCOL['train_seeds'].items() for sample in PROTOCOL['sample_seeds'] for v in ('original','lcc')}
    if len(selected)!=8 or {key(r) for r in selected}!=expected:raise ValueError('protected record coverage')
    result.extend(selected);inventory.append(dict(batch='protected_two_E_early',status='sealed',records=8,sha256=sha(q)))
    return result,inventory
def aggregate(records,templates):
    index={}
    expected={(d,s,a,sample,v) for d,s,a in cells() for sample in PROTOCOL['sample_seeds'] for v in ('original','lcc')}
    for r in records:
        k=key(r)
        if k not in expected:raise ValueError('unexpected identity '+str(k))
        if k in index:raise ValueError('duplicate identity '+str(k))
        index[k]=r
    means=[];paired=[];paired_means=[];cleanup=[];missing=[dict(dataset=d,train_seed=s,arm=a,sample_seed=sample,version=v,reason='not_present_or_failed') for d,s,a,sample,v in sorted(expected-set(index))]
    for d in PROTOCOL['train_seeds']:
        for version in ('original','lcc'):
            for group in ('raw','absolute_error','distribution_error'):
                metrics=set(templates.get(d,{}).get(group,{}))
                for r in records:
                    if r['dataset']==d:metrics.update(r.get(group,{}))
                for metric in sorted(metrics):
                    def value(seed,arm,sample,v=version):return index.get((d,seed,arm,sample,v),{}).get(group,{}).get(metric)
                    for dd,seed,arm in cells():
                        if dd!=d:continue
                        vals=[value(seed,arm,s) for s in PROTOCOL['sample_seeds']];m=mean(vals)
                        means.append(dict(dataset=d,train_seed=seed,arm=arm,version=version,group=group,metric=metric,value=m,reason=None if m is not None else 'both_samples_required_and_defined'))
                        if version=='lcc':
                            for s in PROTOCOL['sample_seeds']:cleanup.append(dict(dataset=d,train_seed=seed,arm=arm,sample_seed=s,group=group,metric=metric,difference=difference(value(seed,arm,s),value(seed,arm,s,'original'))))
                    for dd,seed,left,right in comparisons():
                        if dd!=d:continue
                        vals=[]
                        for s in PROTOCOL['sample_seeds']:
                            delta=difference(value(seed,left,s),value(seed,right,s));vals.append(delta)
                            paired.append(dict(dataset=d,train_seed=seed,sample_seed=s,contrast=left+'-'+right,version=version,group=group,metric=metric,difference=delta,reason=None if delta is not None else 'missing_or_undefined_pair'))
                        m=mean(vals);paired_means.append(dict(dataset=d,train_seed=seed,contrast=left+'-'+right,version=version,group=group,metric=metric,difference=m,reason=None if m is not None else 'both_pairs_required_and_defined'))
    return dict(cell_means=means,paired=paired,paired_means=paired_means,lcc_minus_original=cleanup,missing_identities=missing,comparison_status='complete' if not missing else 'incomplete')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--new-summary',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    target=pathlib.Path(a.output)
    if target.exists():raise FileExistsError('new report only')
    new=read(a.new_summary)
    if new.get('source_sha256')!=source_binding()[1]:raise ValueError('new summary source binding')
    local=[r for r in new['records'] if r.get('batch')=='new_six']
    external,inventory=external_records();records=local+external
    templates={}
    for r in records:templates.setdefault(r['dataset'],{g:r.get(g,{}) for g in ('raw','absolute_error','distribution_error')})
    report=dict(new_batch_status=new['status'],records=records,external_inventory=inventory,new_summary_sha256=sha(a.new_summary),**aggregate(records,templates))
    target.write_text(json.dumps(report,indent=2,allow_nan=False));return 0
if __name__=='__main__':raise SystemExit(main())
