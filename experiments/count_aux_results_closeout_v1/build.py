"""Read saved CPU results only. No torch/model/training/sampling imports.

Usage: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B build.py --output NEW_DIR
Original evidence is read only. New directories must not already exist.
"""
import argparse, collections, csv, datetime, hashlib, importlib.util, itertools
import json, math, os, pathlib, shutil, sys, time

REPO = pathlib.Path('/data2/lyh/gnn_project/DiHug')
SOURCE = REPO/'research_branches/count_aux_length_run/experiments/typed_structure_count_aux_length_v1'
MAIN = REPO/'results/count_aux_length_20261029T231348'
EARLY = REPO/'results/count_aux_early_sampling_20261030T040515'
IMDB = REPO/'results/imdb_count_aux_u6000_20261030T050209'
STAGE = REPO/'results/count_aux_stage500_20261029T205144'
SEEDS = {'IMDB':[2026102911,2026102931], 'PubMed':[2026102912,2026102932]}
STAGES = {'IMDB':[3000,6000], 'PubMed':[1000,2000,3000,6000]}
SAMPLES = [2026102821,2026102822]
NAMES = {'IMDB':['Director','Actor','Movie','Genre'], 'PubMed':['Gene','Disease','Chemical','Species']}
KEYS = ('dataset','train_seed','arm','checkpoint','sample_seed','version')

def read(p): return json.loads(pathlib.Path(p).read_text())
def save(p,x):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def csvwrite(p, rows):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:p.write_text('');return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def ident(r):return tuple(r[k] for k in KEYS)
def jobpath(r):
    return pathlib.Path(f"{r['dataset']}_s{r['train_seed']}_{r['arm']}/u{r['checkpoint']}/ema_0.99/sample_{r['sample_seed']}")
def tag(r):return f"{r['dataset']}_s{r['train_seed']}_{r['arm']}_u{r['checkpoint']}_ema_0.99_{r['sample_seed']}"
def leaves(x,p=''):
    if isinstance(x,dict):
        for k,v in x.items():yield from leaves(v,p+'.'+str(k) if p else str(k))
    elif isinstance(x,list):
        for i,v in enumerate(x):yield from leaves(v,p+f'[{i}]')
    else:yield p,x
def strict_mean(values):
    return sum(values)/len(values) if values and all(x is not None and math.isfinite(float(x)) for x in values) else None
def equivalent(a,b,path='',differences=None):
    """Identity/hash/integer values exact; saved CPU float sums within 1e-12.

    This only audits duplicate saved reports, never changes a metric. Every
    non-bitwise float pair is retained for inspection; the first copy wins.
    """
    if a==b:return True
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(equivalent(a[k],b[k],path+'.'+str(k),differences) for k in a)
    if isinstance(a,list) and isinstance(b,list):
        return len(a)==len(b) and all(equivalent(x,y,path+f'[{i}]',differences) for i,(x,y) in enumerate(zip(a,b)))
    if isinstance(a,float) and isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12):
        if differences is not None:differences.append(dict(field=path,first=a,duplicate=b,absolute_difference=abs(a-b)))
        return True
    return False
def task_identity(task):
    j=task['job'];d,s,a=(j[k] for k in ('dataset','train_seed','arm'))
    if d not in SEEDS or s not in SEEDS[d] or a not in ('CE','CE_COUNT'):
        raise ValueError('Unknown task identity; do not fill dataset by assumption')
    expected=f'{d}_s{s}_{a}'
    if j.get('id',expected)!=expected or j.get('directory',expected)!=expected:
        raise ValueError('Task identity/path disagreement')
    return d,s,a

def collect(out):
    sources=[(MAIN,MAIN/'run/unified_evaluation.json'),(EARLY,EARLY/'evaluation.json'),(IMDB,IMDB/'evaluation.json')]
    index={};origins={};duplicates=[];diversity={}
    for root,p in sources:
        doc=read(p)
        for r in doc['records']:
            k=ident(r)
            if r['dataset'] not in SEEDS or r['train_seed'] not in SEEDS[r['dataset']] or r['checkpoint'] not in STAGES[r['dataset']]:raise ValueError(k)
            if k in index:
                # The batch tag may differ; all metric values and graph hashes must agree.
                for f in ('statistics','comparison','raw','reference','absolute_error','distribution_error','graph_source_sha256','evaluation_source_sha256'):
                    changes=[]
                    if not equivalent(index[k][f],r[f],f,changes):raise ValueError(('duplicate data conflict',k,f))
                    if changes:duplicates.append(dict(identity=k,source=str(p),numeric_roundoff=changes,first_values_retained=True))
                duplicates.append(dict(identity=k,source=str(p),identity_and_hash_exact=True))
            else:index[k]=r;origins[k]=root
        for row in doc['diversity']:
            j=row['job'];k=(j['dataset'],j['train_seed'],j['arm'],j['checkpoint'],row['version'])
            if k in diversity:
                if row['metrics'] is None:
                    duplicates.append(dict(identity=k,source=str(p),historical_pending_placeholder=True,completed_record_retained=diversity[k]['metrics'] is not None));continue
                if diversity[k]['metrics'] is None:diversity[k]=row;continue
                changes=[]
                if not equivalent(diversity[k]['metrics'],row['metrics'],'diversity',changes):raise ValueError(('diversity conflict',k))
                if changes:duplicates.append(dict(identity=k,source=str(p),numeric_roundoff=changes,first_values_retained=True))
            else:diversity[k]=row
    expected={(d,s,a,u,z,v) for d in SEEDS for s,a,u,z,v in itertools.product(SEEDS[d],('CE','CE_COUNT'),STAGES[d],SAMPLES,('original','lcc'))}
    missing=[dict(zip(KEYS,k),reason='not_found_or_incomplete') for k in sorted(expected-index.keys())]
    if set(index)-expected:raise ValueError('unexpected records')
    inventory=[];retention=[];parent_receipts=[]
    for k,r in sorted(index.items()):
        if r['version']!='original':continue
        root=origins[k];src=root/'run'/jobpath(r);b=read(src/'binding.json');j=b['job']
        for f in KEYS[:-1]:
            if j[f]!=r[f]:raise ValueError(('graph binding identity',k,f))
        if b['readout']!='ema_0.99' or j['readout']!='ema_0.99':raise ValueError('not EMA0.99')
        actual=sha(src/'graph.npz')
        if actual!=b['graph_sha256'] or actual!=r['graph_source_sha256']:raise ValueError('graph hash mismatch')
        summary=read(src/'summary.json')
        if summary['status']!='complete':raise ValueError('incomplete saved chain')
        # Check the sidecar identity/hash, without loading any tensor/checkpoint.
        cell=src.parents[2];side=cell/f"readouts_u{r['checkpoint']}.json"
        if side.exists():
            rr=read(side)
            for f in ('dataset','train_seed','arm'):
                if rr[f]!=r[f]:raise ValueError('readout sidecar identity')
            if rr['updates']!=r['checkpoint'] or rr['files']['ema_0.99']['sha256']!=b['model_sha256']:raise ValueError('readout hash/count')
            if rr['checkpoint_sha256']!=b['checkpoint_sha256']:raise ValueError('checkpoint sidecar hash')
            parent_receipts.append(dict(identity=k,source=str(side),sha256=sha(side)))
            copy=out/'run'/jobpath(r).parents[2]/side.name;copy.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(side,copy)
        # Early export has a separate saved verifier; it was already fully audited.
        elif root==EARLY:
            p=root/'preflight.json';parent_receipts.append(dict(identity=k,source=str(p),sha256=sha(p),note='early export readout binding audited in saved preflight and independent audit'))
        else:raise ValueError(('missing readout sidecar',str(side)))
        dst=out/'run'/jobpath(r);dst.mkdir(parents=True,exist_ok=False)
        for name in ('graph.npz','binding.json','summary.json','metrics.json'):
            if (src/name).exists():shutil.copy2(src/name,dst/name)
        lsrc=root/'run/derived_lcc_evaluation'/tag(r);ldst=out/'run/derived_lcc_evaluation'/tag(r)
        shutil.copytree(lsrc,ldst)
        l=index[k[:-1]+('lcc',)];s,ls=r['statistics'],l['statistics']
        retention.append(dict(**{f:r[f] for f in KEYS[:-1]},nodes=s['nodes'],lcc_nodes=ls['nodes'],node_retention=ls['nodes']/s['nodes'],edges=s['edges'],lcc_edges=ls['edges'],edge_retention=ls['edges']/s['edges'] if s['edges'] else None,
                              **{f'type_{t}_retention':ls['typed_degree'][t]['count']/s['typed_degree'][t]['count'] if s['typed_degree'][t]['count'] else None for t in NAMES[r['dataset']]}))
        inventory.append(dict(**{f:r[f] for f in KEYS[:-1]},model='E-earlyCN',readout='ema_0.99',lambda_count=.001 if r['arm']=='CE_COUNT' else 0,
                              loss_formula='sqrt_target_pseudo_huber_count_v2',graph_sha256=actual,weight_sha256=b['model_sha256'],checkpoint_sha256=b['checkpoint_sha256'],original_graph=str(src/'graph.npz'),delivered_graph=str(pathlib.Path('run')/jobpath(r)/'graph.npz'),source_sha256=b['source_sha256'],binding_sha256=sha(src/'binding.json'),status='complete'))
        inventory[-1].update(weight_file=str(cell/f"readout_u{r['checkpoint']}_ema_0.99.pt"),checkpoint_file=str((MAIN if root==EARLY else root)/'run'/jobpath(r).parents[2]/f"checkpoint_u{r['checkpoint']}.pt"),readout_receipt=str(side),readout_receipt_sha256=sha(side))
    for d,root in [('IMDB',IMDB),('PubMed',EARLY)]:
        shutil.copy2(root/f'run/derived_lcc_evaluation/{d}_reference.json',out/f'run/derived_lcc_evaluation/{d}_reference.json')
        (out/'reference').mkdir(exist_ok=True);shutil.copy2(root/f'reference/{d}.npz',out/f'reference/{d}.npz')
    save(out/'material_inventory.json',inventory);csvwrite(out/'tables/material_inventory.csv',inventory)
    save(out/'duplicate_reuse_checks.json',duplicates);save(out/'readout_binding_receipts.json',parent_receipts)
    csvwrite(out/'tables/lcc_retention.csv',retention)
    return list(index.values()),list(diversity.values()),missing

def aggregate(records):
    sys.path.insert(0,str(SOURCE));C=module(SOURCE/'cross_batch.py','saved_cross_batch')
    C.PROTOCOL=dict(train_seeds=SEEDS,arms=['CE','CE_COUNT'],sample_seeds=SAMPLES)
    C.generation_checkpoints=lambda d:STAGES[d]
    C.jobs=lambda:dict(sampling=[dict(dataset=d,train_seed=s,arm=a,checkpoint=u,sample_seed=z) for d in SEEDS for s,a,u,z in itertools.product(SEEDS[d],('CE','CE_COUNT'),STAGES[d],SAMPLES)])
    templates={d:next(r for r in records if r['dataset']==d) for d in SEEDS}
    result=C.aggregate(records,templates)
    # Explicit total 1000->6000 / 3000->6000 deltas complement adjacent-stage tables.
    idx={ident(r):r for r in records};total=[]
    for d in SEEDS:
        for s,a,z,v in itertools.product(SEEDS[d],('CE','CE_COUNT'),SAMPLES,('original','lcc')):
            lo=idx.get((d,s,a,3000,z,v),{});hi=idx.get((d,s,a,6000,z,v),{})
            for g in ('raw','absolute_error','distribution_error'):
                for m in sorted(set(lo.get(g,{}))|set(hi.get(g,{}))):
                    x,y=lo.get(g,{}).get(m),hi.get(g,{}).get(m)
                    total.append(dict(dataset=d,train_seed=s,arm=a,sample_seed=z,version=v,group=g,metric=m,earlier=3000,later=6000,difference=y-x if x is not None and y is not None else None,reason=None if x is not None and y is not None else 'missing_or_undefined_pair'))
    result['total_length_paired']=total;return result

def export_tables(out,result):
    scalars=[];nested=[];undefined=[];catalog=[]
    for r in result['records']:
        key={k:r[k] for k in KEYS}
        for g in ('raw','absolute_error','distribution_error'):
            for m,v in r[g].items():
                ref=r['reference'].get(m) if g!='distribution_error' else 0
                reason=r['undefined'].get(g,{}).get(m) if v is None else None
                scalars.append(dict(**key,group=g,metric=m,value=v,reference=ref,reason=reason))
                if v is None:undefined.append(dict(**key,group=g,metric=m,reason=reason or 'undefined_or_unavailable'))
        for g in ('statistics','comparison'):
            for p,v in leaves(r[g]):nested.append(dict(**key,group=g,field=p,value=v,reason='undefined_or_unavailable' if v is None else None))
    csvwrite(out/'tables/per_graph_all_scalar_metrics.csv',scalars);csvwrite(out/'tables/per_graph_all_detail_fields.csv',nested)
    csvwrite(out/'tables/undefined_values.csv',undefined)
    for d,root in [('IMDB',IMDB),('PubMed',EARLY)]:
        for row in csv.DictReader((root/'tables/metric_catalog.csv').open()):catalog.append(dict(dataset=d,**row))
    csvwrite(out/'tables/metric_catalog.csv',catalog)
    for f in ('cell_means','paired','paired_means','length_paired','length_means','lcc_minus_original','total_length_paired'):csvwrite(out/f'tables/{f}.csv',result[f])
    save(out/'tables/sample_diversity.json',result['diversity']);save(out/'tables/missing_identities.json',result['missing_identities'])
    # Reuse the already-audited detailed vector/table exporter for each dataset.
    all_files=collections.defaultdict(list)
    for d,root in [('IMDB',IMDB),('PubMed',EARLY)]:
        folder=out/'by_dataset'/d;(folder/'tables').mkdir(parents=True)
        save(folder/'evaluation.json',dict(records=[r for r in result['records'] if r['dataset']==d]))
        refdst=folder/'run/derived_lcc_evaluation';refdst.mkdir(parents=True)
        shutil.copy2(out/f'run/derived_lcc_evaluation/{d}_reference.json',refdst/f'{d}_reference.json')
        (folder/'reference').mkdir();shutil.copy2(out/f'reference/{d}.npz',folder/f'reference/{d}.npz')
        m=module(root/'detail_tables.py','detail_'+d);m.OUT=folder;m.main()
        for p in (folder/'tables').glob('*.csv'):
            rows=list(csv.DictReader(p.open()))
            if any(r.get('dataset',d)!=d for r in rows):raise ValueError('detail table dataset mismatch')
            all_files[p.name].extend(dict(r,dataset=d) for r in rows)
        shutil.copy2(root/'METRIC_DIRECTORY.md',folder/'METRIC_DIRECTORY.md')
    for f,rows in all_files.items():csvwrite(out/'tables'/f,rows)

def logs(out):
    validation={};training={};evidence=[];bindings=[]
    for d in SEEDS:
        for s,a in itertools.product(SEEDS[d],('CE','CE_COUNT')):
            cell=f'{d}_s{s}_{a}';roots=[STAGE,MAIN]+([IMDB] if d=='IMDB' else [])
            for root in roots:
                folder=root/'run'/cell
                for filename in ('validation.json','training.jsonl','summary.json','restore_receipt.json'):
                    p=folder/filename
                    if not p.exists():continue
                    dest=out/'saved_logs'/root.name/cell/filename;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
                    evidence.append(dict(source=str(p),delivered=str(dest.relative_to(out)),sha256=sha(p),bytes=p.stat().st_size))
                p=folder/'validation.json'
                if p.exists():
                    for entry in read(p):
                        for kind in ('raw','ema_0.99'):
                            for row in entry.get(kind,{}).get('rows',[]):
                                k=(d,s,a,entry['updates'],kind,row['t'])
                                if k in validation and row!=validation[k]:raise ValueError(('validation duplicate conflict',k))
                                validation[k]=row
                p=folder/'training.jsonl'
                if p.exists():
                    for line in p.open():
                        r=json.loads(line);k=(d,s,a,r['update'])
                        if k in training and r!=training[k]:raise ValueError(('training duplicate conflict',k))
                        training[k]=r
    vals=[];families=[];input_ids=collections.defaultdict(set)
    for k,r in sorted(validation.items()):
        d,s,a,u,kind,t=k;id=dict(dataset=d,train_seed=s,arm=a,update=u,readout=kind,t=t,input_sha256=r['input_sha256'],state_source='fixed_forward_noise')
        input_ids[(d,s,t)].add(r['input_sha256'])
        vals.append(dict(**id,ce=r['ce_sum_over_S'],auxiliary=r.get('auxiliary'),weighted_auxiliary=r.get('weighted_auxiliary'),shadow_weighted_auxiliary=r.get('shadow_weighted_auxiliary'),total_loss=r.get('total_loss'),expected_edges=sum(r['counts']['expected_edges']),target_edges=sum(r['counts']['target_edges'])))
        for f in r['family']:
            family=f['family'];c=r['counts']
            families.append(dict(**id,**f,expected_edges_fp64=c['expected_edges'][family],target_edges_fp64=c['target_edges'][family],signed_error_fp64=c['signed_error'][family],normalized_penalty=c['normalized_penalty'][family],standardized_residual=c['standardized_residual'][family]))
    curves=[];rel=[]
    for (d,s,a,u),r in sorted(training.items()):
        id=dict(dataset=d,train_seed=s,arm=a,update=u)
        curves.append(dict(**id,ce=r['ce'],auxiliary=r['auxiliary'],weighted_auxiliary=r['weighted_auxiliary'],loss=r['loss'],gradient_norm=r.get('gradient_norm'),seconds=r['total_update_seconds'],input_sha256=json.dumps(r['input_sha256']),t=json.dumps(r['t'])))
        for i,x in enumerate(r['inputs']):
            for f,(mu,m,delta,penalty) in enumerate(zip(x['counts']['expected_edges'],x['counts']['target_edges'],x['counts']['signed_error'],x['counts']['normalized_penalty'])):
                rel.append(dict(**id,input=i,t=x['t'],family=f,mu=mu,m=m,residual=delta,normalized_penalty=penalty))
    pair_checks=[]
    for d in SEEDS:
        for s in SEEDS[d]:
            ra={u:r['input_sha256'] for (dd,ss,a,u),r in training.items() if (dd,ss,a)==(d,s,'CE')}
            rb={u:r['input_sha256'] for (dd,ss,a,u),r in training.items() if (dd,ss,a)==(d,s,'CE_COUNT')}
            pair_checks.append(dict(dataset=d,train_seed=s,updates=len(ra),paired_inputs_exact=ra==rb,complete_updates=set(ra)==set(range(1,6001))))
    save(out/'validation_panel.json',vals);csvwrite(out/'tables/validation_total.csv',vals);csvwrite(out/'tables/validation_relations.csv',families)
    csvwrite(out/'tables/training_curve.csv',curves);csvwrite(out/'tables/training_relations.csv',rel)
    save(out/'log_bindings.json',evidence);save(out/'input_pairing.json',pair_checks)
    save(out/'validation_input_consistency.json',[dict(dataset=d,train_seed=s,t=t,unique_inputs=len(v),hashes=sorted(v),consistent=len(v)==1) for (d,s,t),v in input_ids.items()])
    return vals,curves

def plots(out,result,validation,training):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fields=[('absolute_error','edges','Edge abs error'),('distribution_error','main_incident4_W1','Degree W1'),('distribution_error','main_joint_degree_TV','Joint-degree TV'),('absolute_error','triangles','Triangle abs error'),('distribution_error','triangle_composition_TV','Triangle composition TV'),('raw','type_Chemical_low_le1_fraction','Chemical low-degree fraction'),('absolute_error','family1_edges','GD edge abs error'),('absolute_error','family5_edges','DC edge abs error'),('absolute_error','type_Chemical_maximum','Chemical hub abs error'),('distribution_error','MP3_composition_TV','MP3 composition TV')]
    for d in SEEDS:
        fs=fields if d=='PubMed' else fields[:5]+[('absolute_error','type_Genre_maximum','Genre max-degree error'),('absolute_error','type_Genre_second','Genre second-degree error'),('absolute_error','triangle_(0, 1, 2)','DAM abs error'),('absolute_error','triangle_(0, 1, 1)','DAA abs error'),('distribution_error','MP3_composition_TV','MP3 composition TV')]
        for version in ('original','lcc'):
            fig,axes=plt.subplots(len(fs),2,figsize=(11,22),squeeze=False)
            for col,s in enumerate(SEEDS[d]):
                for row,(g,m,label) in enumerate(fs):
                    ax=axes[row,col]
                    for a in ('CE','CE_COUNT'):
                        data=[x for x in result['cell_means'] if (x['dataset'],x['train_seed'],x['arm'],x['version'],x['group'],x['metric'])==(d,s,a,version,g,m)]
                        data.sort(key=lambda x:x['checkpoint']);ax.plot([x['checkpoint'] for x in data],[x['value'] if x['value'] is not None else float('nan') for x in data],marker='o',label=a)
                    ax.set_title(f'{label}; seed {s}');ax.grid(alpha=.3);ax.legend(fontsize=8)
            fig.suptitle(f'{d} {version}: actual generated graphs; mean of two samples');fig.tight_layout(rect=(0,0,1,.98))
            for ext in ('png','pdf'):fig.savefig(out/f'{d}_generation_{version}.{ext}',dpi=130)
            plt.close(fig)
        fig,axes=plt.subplots(3,2,figsize=(12,10),squeeze=False)
        for col,s in enumerate(SEEDS[d]):
            for a in ('CE','CE_COUNT'):
                rr=[x for x in training if (x['dataset'],x['train_seed'],x['arm'])==(d,s,a)]
                for axis,metric in zip(axes[:2,col],('ce','weighted_auxiliary')):
                    # Fixed 100-update window, descriptive smoothing only; raw curves remain complete.
                    chunks=[rr[i:i+100] for i in range(0,len(rr),100)]
                    axis.plot([v[-1]['update'] for v in chunks],[sum(x[metric] for x in v)/len(v) for v in chunks],label=a);axis.set_title(f'{metric}; seed{s}');axis.legend();axis.grid(alpha=.3)
                for kind,style in [('raw','--'),('ema_0.99','-')]:
                    r=[x for x in validation if (x['dataset'],x['train_seed'],x['arm'],x['readout'],x['t'])==(d,s,a,kind,500)]
                    axes[2,col].plot([x['update'] for x in r],[x['ce'] for x in r],style,marker='o',label=f'{a} {kind}')
                axes[2,col].set_title('Fixed t=500 validation CE');axes[2,col].legend(fontsize=7);axes[2,col].grid(alpha=.3)
        fig.tight_layout();fig.savefig(out/f'{d}_training_validation.png',dpi=140);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=pathlib.Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);begin=time.time()
    (out/'tables').mkdir();(out/'evidence').mkdir()
    save(out/'closeout_started.json',dict(epoch=begin,time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),scope='CPU reports only',new_model_forwards=0,new_training_updates=0,new_chains=0))
    records,diversity,missing=collect(out);result=dict(records=records,diversity=diversity,**aggregate(records))
    if result['missing_identities']!=missing:raise ValueError('independent inventory mismatch')
    save(out/'evaluation.json',result);export_tables(out,result);validation,training=logs(out);plots(out,result,validation,training)
    for name in ('evaluation_fields.py','cross_batch.py','lcc_eval/metrics.py'):
        dest=out/'frozen_evaluation_source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/name,dest)
    shutil.copytree(SOURCE/'lcc_eval/vendor',out/'frozen_evaluation_source/lcc_eval/vendor')
    shutil.copy2(SOURCE/'formula.json',out/'formula.json');shutil.copy2(SOURCE/'METRICS.md',out/'METRICS.md');shutil.copy2(EARLY/'metric_definition_reference.md',out/'metric_definition_reference.md')
    for root,names in [(MAIN,['delivery_failure.json','ticket_ledger.json','run/launcher_summary.json','run/stage_status.json']),(EARLY,['completion.json','final_delivery.json','independent_audit.json','preflight.json','ZIP_SHA256.txt']),(IMDB,['completion.json','independent_audit.json','ZIP_SHA256.txt'])]:
        for name in names:
            dest=out/'evidence'/root.name/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,dest)
    save(out/'build_receipt.json',dict(status='complete',seconds=time.time()-begin,records=len(records),unique_original_graphs=sum(r['version']=='original' for r in records),missing=missing,torch_imported='torch' in sys.modules,mathematical_evaluation_changed=False,new_training_updates=0,new_model_forwards=0,new_chains=0))
    print(json.dumps(read(out/'build_receipt.json')))
if __name__=='__main__':main()
