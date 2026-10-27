"""Single-GPU engineering diagnostics, zero optimizer steps, zero graph draws."""
import argparse,gc,json,os,pathlib,resource,signal,subprocess,time,traceback
START=time.time()
import numpy as np
import torch
from prepare import ROOT,load_case,sha,save
from benchmark import compare
from model_variants import StructuralDenoiser,copy_common
from hsgd.bw_preflight.common import train_input,update,plan_for,schedule,environment
from hsgd.computational import FrozenTypedContext
from hsgd.kernel import MarginalKernel

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--ledger',required=True);args=p.parse_args()
    ledger=json.loads(pathlib.Path(args.ledger).read_text());cutoff=min(START+300,ledger['activation_epoch']+2700-180)
    out=pathlib.Path(args.output).resolve()
    if out.exists() or out.is_relative_to(ROOT.resolve()) or ROOT.resolve().is_relative_to(out):raise ValueError('new separate output required')
    out.mkdir(parents=True);summary={'status':'running','optimizer_steps':0,'generated_graphs':0,'cases':{}}
    def check():
        if time.time()>cutoff:raise TimeoutError('forward budget')
        if (out/'STOP').exists():raise InterruptedError('STOP')
    def stop(s,f):raise InterruptedError(str(s))
    for s in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(s,stop)
    signal.signal(signal.SIGALRM,lambda s,f:(_ for _ in ()).throw(TimeoutError('alarm')));signal.alarm(max(1,int(cutoff-time.time())))
    try:
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True,warn_only=False)
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('one selected GPU required')
        summary['environment']=environment('cuda');save(out/'freeze.json',{str(x.relative_to(ROOT)):sha(x) for x in ROOT.rglob('*') if x.is_file() and '__pycache__' not in str(x)})
        for dataset in ['IMDB','PubMed']:
            c,cfg=load_case(dataset);entry=cfg['datasets'][dataset]
            if sha(ROOT/entry['basis_file'])!=entry['basis_sha256']:raise ValueError('basis')
            basis=np.load(ROOT/entry['basis_file'],allow_pickle=False)['basis'];torch.manual_seed(cfg['seed']);anchor=StructuralDenoiser(c,cfg,'A')
            inp=train_input(c,1000,20,cfg['seed'],50,np.random.default_rng(2026102712),force_t=500)
            plan=plan_for(c,[2026102712,500,61],20);state=np.zeros(c['space'].total,dtype=np.uint8)
            f,loc=c['space'].encode(inp['edges']);state[c['space'].offsets[f]+loc]=1;kernel=MarginalKernel(c['prior'],schedule(1000))
            for arm in 'ABC':
                check();name=dataset+'_'+arm;folder=out/name;folder.mkdir();row={'status':'running'};summary['cases'][name]=row
                torch.manual_seed(cfg['seed']);model=StructuralDenoiser(c,cfg,arm,basis);copy_common(anchor,model)
                if name!='IMDB_A':
                    # Evidence only; failure is retained, no optimizer updates or permission escalation follows.
                    kw={k:v for k,v in cfg['optimizer'].items() if k!='type'};opt=torch.optim.AdamW(model.parameters(),**kw);cpu={}
                    update(model,opt,c,inp,1000,'cpu',do_step=False,diagnostics=cpu)
                    model.to('cuda');opt=torch.optim.AdamW(model.parameters(),**kw);gpu={}
                    update(model,opt,c,inp,1000,'cuda',do_step=False,diagnostics=gpu)
                    gate={k:compare(cpu[k],gpu[k]) for k in ['logits','p0','loss']}
                    gradients={k:compare(v,gpu['gradients'][k]) for k,v in cpu['gradients'].items()}
                    gate['gradient']={'passes':all(v['passes'] for v in gradients.values()),'max_abs':max(v['max_abs'] for v in gradients.values()),
                                      'violations':sum(v['violations'] for v in gradients.values())}
                    row['gate']=gate;save(folder/'gradients_gate.json',gradients);del opt,cpu,gpu
                else:
                    model.to('cuda');row['gate']='previous actual IMDB_A gate retained; not repeated'
                model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0);model.eval()
                rows=[];first=None
                for repeat in range(2):
                    check();torch.cuda.synchronize();start=time.perf_counter();feature=model.feature_seconds;encode=model.encoding_seconds;transfer=model.transfer_seconds;calls=model.feature_calls
                    token=('read_only',repeat);ctx=FrozenTypedContext(model,c['space'].types,inp['edges'],inp['edge_families'],token)
                    vals=np.full(len(state),np.nan);q=vals.copy();seen=np.zeros(len(state),np.uint8);network=0.;posterior=0.
                    try:
                        with torch.no_grad():
                            for block in range(20):
                                check();ids,ff,qe=plan.block(block);tick=time.perf_counter()
                                pp=ctx.predict(qe,ff,.5,token).cpu().numpy().astype(np.float64);torch.cuda.synchronize();network+=time.perf_counter()-tick
                                tick=time.perf_counter();q[ids]=kernel.reverse(pp,state[ids],ff,499,500);posterior+=time.perf_counter()-tick;vals[ids]=pp;seen[ids]+=1
                    finally:ctx.close()
                    assert np.array_equal(seen,np.repeat(c['active'].astype(np.uint8),c['space'].counts))
                    if first is not None:np.testing.assert_allclose(vals,first,atol=2e-6,rtol=1e-5,equal_nan=True)
                    first=vals.copy();torch.cuda.synchronize()
                    rows.append(dict(repeat=repeat,wall_seconds=time.perf_counter()-start,covered=int(seen.sum()),blocks=20,
                         feature_seconds=model.feature_seconds-feature,encoding_seconds=model.encoding_seconds-encode,
                         transfer_seconds=model.transfer_seconds-transfer,feature_calls=model.feature_calls-calls,
                         query_network_including_features_seconds=network,posterior_seconds=posterior))
                np.savez_compressed(folder/'full_query.npz',state=state,p0=vals,q=q,coverage=seen)
                row.update(status='complete_forward_only',timings=rows,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                    device_total_occupancy_snapshot=subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
                save(out/'summary.json',summary);print(name,rows[-1]['wall_seconds'],flush=True)
                del model;gc.collect();torch.cuda.empty_cache()
        summary['status']='completed_forward_evidence_only'
    except BaseException as exc:
        summary.update(status='incomplete' if isinstance(exc,(TimeoutError,InterruptedError)) else 'failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        signal.alarm(0);summary['seconds']=time.time()-START;save(out/'summary.json',summary)
if __name__=='__main__':main()
