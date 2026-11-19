"""Read-only two-position response along an already saved displacement."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,copy,importlib.util,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def mod(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
base=mod('response_base',ROOT/'research_branches/exact_feedback_updates_20261110/worker.py')
rt,torch,np,g=base.rt,base.torch,base.np,base.g
HardScore,Exact,remaining=base.HardScore,base.Exact,base.remaining
OUT=ROOT/'results/displacement_response_20261110'
OLD=ROOT/'results/exact_feedback_updates_20261110'
def main():
    row=json.loads((OUT/'protocol.json').read_text())['tasks'][int(sys.argv[1])]
    folder=OUT/f"{row['dataset']}_{row['seed']}";folder.mkdir(exist_ok=False)
    source=Path(row['response_source']);start=time.time();receipt=dict(forwards=0,backwards=0,updates=0,status='running')
    try:
        rt.setup();assert torch.cuda.is_available() and torch.cuda.device_count()==1
        for p,h in json.loads((OUT/'bindings.json').read_text()).items():assert rt.sha(p)==h,p
        z,cfg,case,m,o,e,rng,b=rt.load(row)
        end=torch.load(source/'displacement_complete.pt',map_location='cpu',weights_only=False)['state']
        assert end['updates']==2001 and z['updates']==2000
        train=torch.load(source/'update_input.pt',map_location='cpu',weights_only=False)
        fixed=torch.load(OLD/folder.name/'evaluation_inputs.pt',map_location='cpu',weights_only=False)['inputs']
        assert rt.input_hash([train])==row['diagnostic_input_hash']
        names=set(dict(m.named_parameters()));buffers=set(dict(m.named_buffers()))
        assert names|buffers==set(z['model'])==set(end['model'])
        for n in buffers:assert torch.equal(z['model'][n],end['model'][n]),'buffer changed '+n
        original=g.digest(b.checkpoint(z['binding']));records=[]
        rt.save(folder/'environment.json',g.common.environment('cuda'))
        for alpha in (.25,.5):
            rt.restore(z,m,o,e,rng)
            with torch.no_grad():
                for n,p in m.named_parameters():p.copy_((z['model'][n].double()+alpha*(end['model'][n].double()-z['model'][n].double())).to(device=p.device,dtype=p.dtype))
            snap=g.digest(b.checkpoint(z['binding']))
            with b.validation():
                for j,(label,inp,mode) in enumerate([('same',train,True)]+[(f't{x["t"]}',x,False) for x in fixed]):
                    assert time.time()<json.loads((OUT/'window.json').read_text())['science_stop_epoch']
                    m.train(mode);token=('response',alpha,j);ctx=g.FrozenTypedContext(m,case['space'].types,inp['edges'],inp['edge_families'],token)
                    try:
                        with torch.no_grad():
                            graph,logits,p=ctx.predict(inp['query'],inp['f'],inp['t']/1000,token,details=True)
                            g.common.check_prob_device(p,inp['f'],case,m.prior)
                            terms=rt.supported_binary_ce(logits,torch.as_tensor(inp['labels'],device='cuda'),m.prior[graph.query_families]).cpu().numpy()
                        receipt['forwards']+=1;ix=np.argsort(inp['ids'])
                        np.savez_compressed(folder/f'a{alpha}_{label}.npz',probability=p.cpu().double().numpy()[ix],terms=terms[ix],labels=inp['labels'][ix],families=inp['f'][ix],ids=inp['ids'][ix],query=inp['query'][:,ix])
                        records.append(dict(alpha=alpha,panel=label,t=inp['t'],training_mode=mode,input_hash=rt.input_hash([inp])))
                    finally:ctx.close()
            assert snap==g.digest(b.checkpoint(z['binding'])),'evaluation state changed'
        rt.restore(z,m,o,e,rng);assert original==g.digest(b.checkpoint(z['binding']))
        rt.save(folder/'evaluation.json',records);rt.save(folder/'checks.json',dict(state_preserved=True,optimizer_EMA_RNG_counters_unchanged=True,buffers_equal=True,parameter_interpolation='float64 endpoint difference then FP32 write',normal_CUDA_bitwise_recomputation_not_certified=True))
        receipt['status']='completed'
    except BaseException as exc:
        receipt.update(status='failed',error=repr(exc));(folder/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        receipt.update(seconds=time.time()-start,peak_GPU_allocated_MiB=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_initialized() else 0);rt.save(folder/'receipt.json',receipt)
if __name__=='__main__':main()
