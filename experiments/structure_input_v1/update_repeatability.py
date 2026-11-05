"""Eight isolated updates; memory versus file roundtrip, no training trajectory."""
import argparse, copy, hashlib, itertools, json, time, traceback
from pathlib import Path
from experiments.structure_input_v1 import gpu_engineering as g
torch, np = g.torch, g.np

def cpu(v):
    if torch.is_tensor(v): return v.detach().cpu().clone()
    if isinstance(v,dict): return {k:cpu(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)): return type(v)(cpu(x) for x in v)
    return copy.deepcopy(v)

def compare(a,b):
    rows=[]
    for name,x in a.items():
        y=b[name]
        if x is None or y is None:
            assert x is None and y is None
            continue
        d=(x.double()-y.double()).abs(); flat=int(d.argmax())
        rows.append(dict(name=name,max_abs=float(d.flatten()[flat]),nonzero=int((d!=0).sum()),
                         flat_index=flat,left=float(x.flatten()[flat]),right=float(y.flatten()[flat])))
    return dict(max_abs=max((r['max_abs'] for r in rows),default=0),
                nonzero=sum(r['nonzero'] for r in rows),per_parameter=rows,
                worst=max(rows,key=lambda r:r['max_abs']) if rows else None)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--stop',type=float,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    start=time.time();report=dict(status='running',start_epoch=start,stop_epoch=args.stop,
       source_commit='f47c0a67a5ec269308d97dbd1c9b3b2adb380cef',order=['M1','F1','M2','F2'],
       protocol='Each run restores identical complete boundary; M reuses CPU object, F reloads its saved copy.',runs=[],comparisons=[])
    def emit():g.save(args.output/'result.json',report)
    def check():
        if time.time()>=args.stop:raise TimeoutError('absolute deadline')
    emit();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    report['environment']=g.common.environment('cuda')
    report['deterministic']=torch.are_deterministic_algorithms_enabled()
    report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    tasks=json.loads((args.assets/'results/cn_multiseed_20261104T042546/tasks.json').read_text())
    cfg=next(x['config'] for x in tasks if x['dataset']=='IMDB' and x['train_seed']==2026110411 and x['arm']=='A')
    report['config']=cfg
    data=args.assets/g.SCI.relative_to(g.ROOT)/'package'/cfg['datasets']['IMDB']['file']
    assert hashlib.sha256(data.read_bytes()).hexdigest()==cfg['datasets']['IMDB']['sha256']
    with np.load(data,allow_pickle=False) as z:
        case=g.common.make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
        assert np.array_equal(case['space'].encode(z['edges'].T)[0]+1,z['relations'])
    keys=('model','optimizer','ema','rng','updates')
    try:
        for arm in ('A','candidate'):
            check()
            source=args.assets/'results/structure_input_gpu_20261106T041548/gpu_run_followup'/('IMDB_'+arm+'_engineering_step1.pt')
            base=torch.load(source,map_location='cpu');assert base['updates']==base['ema']['updates']==1
            original_hash=g.digest({k:base[k] for k in keys})
            disk=args.output/(arm+'_roundtrip.pt');torch.save(base,disk)
            model=(g.StructuralDenoiser(case,cfg,'A') if arm=='A' else g.StructureInputA(case,cfg)).cuda()
            opt=torch.optim.AdamW(model.parameters(),**{k:v for k,v in cfg['optimizer'].items() if k!='type'})
            ema=g.ParameterEMA(model);rng=np.random.default_rng(1);binding=base['binding'];records={}
            report.setdefault('bindings',{})[arm]=dict(parent_path=str(source),parent_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),state_sha256=original_hash)
            for label in report['order']:
                check();tick=time.time()
                # load_state_dict may retain tensor references: always give it a fresh copy.
                payload=copy.deepcopy(base) if label[0]=='M' else torch.load(disk,map_location='cpu')
                count=g.restore_payload(payload,model,opt,ema,binding,lambda s:g.rng_restore(s,rng,'cuda'))
                tx=g.TrainingBoundary(model,opt,ema,rng,'cuda',count)
                restored=tx.checkpoint(binding)
                assert g.digest({k:restored[k] for k in keys})==original_hash
                assert g.digest({k:base[k] for k in keys})==original_hash
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                with tx.step(2):
                    inp=g.one_input(case,cfg,2,rng,force_t=500)
                    ih=g.digest({k:inp[k] for k in ('t','ids','f','query','labels','edges','edge_families')})
                    assert ih=='031d73716bef63195f3f6981d30b26886c5add710b0a2567435db30df255a605'
                    observation=g.update(model,opt,case,inp,1000,'cuda',ema,tx.completed,transaction=tx)
                torch.cuda.synchronize()
                named=dict(model.named_parameters())
                grads={n:cpu(p.grad) for n,p in named.items()}
                params={n:cpu(p) for n,p in named.items()}
                delta={n:p-base['model'][n] for n,p in params.items()}
                moments={n:cpu(opt.state[p]) for n,p in named.items()}
                final=tx.checkpoint(binding)
                record=dict(parameters=params,gradients=grads,delta=delta,moments=moments,full_state=final)
                path=args.output/(arm+'_'+label+'.pt');torch.save(record,path);records[label]=record
                row=dict(arm=arm,label=label,exact_boundary=True,input_sha256=ih,
                    result=observation,seconds=time.time()-tick,peak_mib=torch.cuda.max_memory_allocated()/2**20,
                    update_l2=sum(float(v.double().square().sum()) for v in delta.values())**.5,
                    rng_sha256=g.digest(final['rng']),file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                report['runs'].append(row);emit()
            for left,right in itertools.combinations(report['order'],2):
                x,y=records[left],records[right]
                row=dict(arm=arm,left=left,right=right,loss_abs=abs(next(r['result']['loss'] for r in report['runs'] if r['arm']==arm and r['label']==left)-next(r['result']['loss'] for r in report['runs'] if r['arm']==arm and r['label']==right)))
                for field in ('parameters','gradients','delta'):row[field]=compare(x[field],y[field])
                for field in ('exp_avg','exp_avg_sq','step'):
                    row[field]=compare({n:s[field] for n,s in x['moments'].items() if field in s},{n:s[field] for n,s in y['moments'].items() if field in s})
                row['rng_equal']=g.digest(x['full_state']['rng'])==g.digest(y['full_state']['rng'])
                row['complete_state']=g.compare_state(x['full_state'],y['full_state'],cfg['tolerances']['fp32_atol'],cfg['tolerances']['fp32_rtol'])
                w=row['parameters']['worst'];name=w['name'];idx=w['flat_index']
                def value(v):return float(v.flatten()[idx])
                row['worst_coordinate']=dict(name=name,flat_index=idx,shape=list(x['parameters'][name].shape),before=value(base['model'][name]),
                    runs={s:dict(parameter=value(records[s]['parameters'][name]),gradient=value(records[s]['gradients'][name]),
                                actual_delta=value(records[s]['delta'][name]),m=value(records[s]['moments'][name]['exp_avg']),
                                v=value(records[s]['moments'][name]['exp_avg_sq']),step=float(records[s]['moments'][name]['step'])) for s in (left,right)})
                report['comparisons'].append(row)
            del records,model,opt,ema,payload,restored,base,final,record,x,y,named,params,grads,delta,moments
            torch.cuda.empty_cache();emit()
        report['status']='completed'
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc();raise
    finally:
        report['end_epoch']=time.time();report['seconds']=time.time()-start;emit()

if __name__=='__main__':main()
