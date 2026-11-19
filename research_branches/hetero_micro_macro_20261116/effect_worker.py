"""Frozen gradient calibration and paired short training; no policy changes.

All diagnostics and states live under the authorized execution directory.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,traceback,signal,hashlib,importlib.util,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def write(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
visible=os.environ.get('CUDA_VISIBLE_DEVICES','0')
sys.path.insert(0,str(ROOT/'research_branches/analytic_joint_risk_20261113'))
from core import risks,probability_metrics
os.environ['CUDA_VISIBLE_DEVICES']=visible
from objective import make_plan,descriptors,distribution_errors
from gradient_scale import BLOCKS,freeze_coefficients,compose
spec=importlib.util.spec_from_file_location('effect_support',ROOT/'research_branches/one_step_vs_terminal_plan_20261111/update_preflight.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
w,np,torch=s.w,s.np,s.torch
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
TIMES=[1,75,250,550,800,1000]

class Context:
    def __init__(self,out,index):
        self.out=Path(out);self.protocol=json.loads((self.out/'protocol.json').read_text());self.row=self.protocol['parents'][index]
        self.calls=self.backs=0;self.ds=self.row['dataset'];self.seed=self.row['train_seed']
        for p,h in self.protocol['runtime_source_bindings'].items():assert sha(p)==h,p
        self.state=torch.load(self.row['initial_complete'],map_location='cpu',weights_only=False)
        assert sha(self.row['initial_complete'])==self.row['initial_complete_sha256']
        assert self.state['updates']==0 and not self.state['optimizer']['state']
        assert s.state_digest(self.state['model'])==self.row['parent_model_hash']
        z=np.load(self.row['data']);self.ty=z['types'];self.schema=z['schema'];ee=z['edges'];ee=ee if ee.shape[0]==2 else ee.T
        self.case=w.common.make_case(self.ty,ee,tuple(map(tuple,self.schema)))
        self.q=np.array(list(itertools.combinations(range(len(self.ty)),2))).T
        lookup={tuple(r):j for j,r in enumerate(self.schema)}
        self.f=np.array([lookup[tuple(sorted(self.ty[e]))] for e in self.q.T]);self.keymap={tuple(e):j for j,e in enumerate(self.q.T)}
        edges={tuple(sorted(e)) for e in ee.T};self.ref=np.array([tuple(e) in edges for e in self.q.T],float)
        self.prior=np.bincount(self.f,weights=self.ref,minlength=len(self.schema))/np.bincount(self.f,minlength=len(self.schema))
        self.plan=make_plan(self.schema,len(np.unique(self.ty)))
        self.degree=json.loads((ROOT/'results/analytic_joint_risk_20261113'/(self.ds+'_frozen_target.json')).read_text())
        f=json.loads((ROOT/'results/hetero_micro_macro_cpu_20261116_v2'/(self.ds+'_frozen.json')).read_text())
        self.frozen={int(t):{g:dict(scale=torch.tensor(b['scale'],dtype=torch.float64),omega=torch.tensor(b['omega'],dtype=torch.float64),
            target=torch.tensor(np.array(b['target_real'])+1j*np.array(b['target_imag']),dtype=torch.complex128)) for g,b in groups.items()} for t,groups in f.items()}
        self.sigma=json.loads((ROOT/'results/hetero_micro_macro_calibration_20261116/frozen_scales.json').read_text())['sigma_squared'][self.ds]
        self.m=w.StructuralDenoiser(self.case,self.row['config'],'A').cuda();self.m.eval()
        oc={k:v for k,v in self.row['config']['optimizer'].items() if k!='type'};oc['lr']=1e-5
        self.o=torch.optim.AdamW(self.m.parameters(),**oc);self.ema=s.ParameterEMA(self.m,.99);self.rng=np.random.default_rng()
        self.boundary=self.restore(self.state)
    def check(self):
        if time.time()>=self.protocol['science_stop_epoch'] or (self.out/'STOP').exists():raise TimeoutError('science/allocation stop')
    def restore(self,st):
        s.restore_payload(st,self.m,self.o,self.ema,st['binding'],lambda r:s.rng_restore(r,self.rng,'cuda'))
        self.o.zero_grad(set_to_none=True);self.m.eval()
        b=s.TrainingBoundary(self.m,self.o,self.ema,self.rng,'cuda',st['updates'])
        assert s.state_digest(b.checkpoint(st['binding']))==s.state_digest(st)
        return b
    def checkpoint(self):return self.boundary.checkpoint(self.state['binding'])
    def forward(self,inp):
        self.check();lo,pr=w.runtime.predict(self.m,self.ty,inp['edges'],inp['edge_families'],inp['query'],inp['f'],inp['t']/1000,details=True);self.calls+=1
        w.common.check_prob_device(pr,inp['f'],self.case,self.m.prior)
        terms=s.supported_binary_ce(lo,torch.as_tensor(inp['labels'],device='cuda'),self.m.prior[torch.as_tensor(inp['f'],device='cuda')])
        return lo,pr,terms
    def expand(self,pr,inp,grad=False):
        ix=np.array([self.keymap[tuple(sorted(e))] for e in inp['query'].T]);assert len(set(ix))==len(ix)
        np.testing.assert_array_equal(self.f[ix],inp['f']);np.testing.assert_array_equal(self.ref[ix],inp['labels'])
        p=self.prior[self.f].copy();p[ix]=pr.detach().cpu().double().numpy();missing=np.ones(len(p),bool);missing[ix]=False
        assert np.all((p[missing]==0)|(p[missing]==1))
        return torch.tensor(p,dtype=torch.float64,requires_grad=grad),ix
    def blocks(self,tp):
        P=tp.new_zeros((len(self.schema),len(self.ty),len(self.ty)))
        P[self.f,self.q[0],self.q[1]]=tp;P[self.f,self.q[1],self.q[0]]=tp
        err=[x for x in distribution_errors(descriptors(P,self.ty,self.plan),self.ty,self.frozen) if x['block']!='degree']
        _,part=risks(tp,self.ty,self.q,self.f,self.degree)
        err += [dict(type=x['type'],block='degree',error=x['risk']) for x in part]
        losses={b:torch.stack([x['error']/(4*self.sigma[str(x['type'])][b]) for x in err if x['block']==b]).mean() for b in BLOCKS}
        return losses,err
    def inputs(self,split):
        result=[]
        if split=='calibration':
            for j,t in enumerate(TIMES):result.append(w.common.train_input(self.case,1000,1,self.seed,920000+j,np.random.default_rng(np.random.SeedSequence([202611130953,self.seed,j])),force_t=t))
            old=json.loads((ROOT/'results/analytic_joint_model_check_20261113'/(self.row['id']+'.json')).read_text())
            assert [s.input_hash([i]) for i in result]==old['input_hashes']['update']
        elif split=='train':
            for step in range(1,129):
                r=np.random.default_rng(np.random.SeedSequence([2026111611,self.seed,step]));t=int(r.integers(1,1001))
                result.append(w.common.train_input(self.case,1000,1,self.seed,960000+step,r,force_t=t))
        else:
            for j,t in enumerate(TIMES):
                for rep in range(2):result.append(w.common.train_input(self.case,1000,1,self.seed,970000+2*j+rep,np.random.default_rng(np.random.SeedSequence([2026111612,self.seed,j,rep])),force_t=t))
        return result
    def evaluate(self,inputs,coeff,ema=False):
        before=s.state_digest(self.checkpoint());rows=[]
        def run(_):
            with torch.no_grad():
                for inp in inputs:
                    lo,pr,terms=self.forward(inp);tp,ix=self.expand(pr,inp);blocks,errs=self.blocks(tp)
                    ce=float(terms.sum()/self.case['space'].total);lab=inp['labels'];fam=inp['f'];ts=terms.cpu().numpy();pp=pr.cpu().double().numpy();rels=[]
                    for r in range(len(self.schema)):
                        pos=(fam==r)&(lab==1);neg=(fam==r)&(lab==0)
                        rels.append(dict(relation=r,positive_n=int(pos.sum()),negative_n=int(neg.sum()),positive_CE=float(ts[pos].mean()) if pos.any() else None,
                            negative_CE=float(ts[neg].mean()) if neg.any() else None,missed_mass=float((1-pp[pos]).sum()),false_mass=float(pp[neg].sum())))
                    rows.append(dict(t=inp['t'],input_hash=s.input_hash([inp]),CE=ce,positive_CE=float(ts[lab==1].mean()),negative_CE=float(ts[lab==0].mean()),
                        candidate_loss=float(compose(ce,blocks,coeff)),blocks={k:float(v) for k,v in blocks.items()},
                        errors=[dict(type=x['type'],block=x['block'],error=float(x['error'])) for x in errs],relations=rels,
                        structure=probability_metrics(tp.numpy(),self.ty,self.q,self.f,self.ref,len(self.schema))))
        with self.boundary.validation():
            if ema:s.ema_validation(run,self.m,self.ema,boundary=self.boundary)
            else:run(self.m)
        assert before==s.state_digest(self.checkpoint())
        return rows
    def update(self,inp,arm,coeff,step):
        self.check();start=time.time();self.o.zero_grad(set_to_none=True)
        with self.boundary.step(step):
            old_rng=s.state_digest(s.rng_capture(self.rng,'cuda'));lo,pr,terms=self.forward(inp);ce=terms.sum()/self.case['space'].total
            if arm=='calibrated_micro_macro':
                tp,ix=self.expand(pr,inp,True);blocks,_=self.blocks(tp);macro=compose(tp.new_zeros(()),blocks,coeff)
                dp=torch.autograd.grad(macro,tp)[0];assert torch.isfinite(dp).all()
                torch.autograd.backward([ce,pr],[torch.ones_like(ce),dp[ix].to(pr)])
            else:macro=ce.detach()*0;ce.backward()
            self.backs+=1
            assert old_rng==s.state_digest(s.rng_capture(self.rng,'cuda'))
            norm=float(torch.nn.utils.clip_grad_norm_(self.m.parameters(),1,error_if_nonfinite=True))
            self.boundary.mark_optimizer_started();self.o.step();self.ema.update(self.m,step);self.ema.at_update_boundary=True
            assert all(torch.isfinite(v).all() for v in self.m.parameters())
        return dict(step=step,t=inp['t'],input_hash=s.input_hash([inp]),CE=float(ce.detach()),macro=float(macro.detach()),gradient_norm=norm,seconds=time.time()-start)

def calibrate(out):
    rows=[];calls=backs=0;start=time.time()
    for index in range(4):
        c=Context(out,index);before=s.state_digest(c.checkpoint());inputs=c.inputs('calibration');params=list(c.m.parameters())
        for inp in inputs:
            lo,pr,terms=c.forward(inp);ce=terms.sum()/c.case['space'].total;tp,ix=c.expand(pr,inp,True);blocks,_=c.blocks(tp)
            gradients={};norms={}
            for key in ('CE',)+BLOCKS:
                c.check()
                if key=='CE':g=torch.autograd.grad(ce,params,retain_graph=True,allow_unused=True)
                else:
                    dp=torch.autograd.grad(blocks[key],tp,retain_graph=key!=BLOCKS[-1])[0]
                    assert torch.isfinite(dp).all()
                    g=torch.autograd.grad(pr,params,grad_outputs=dp[ix].to(pr),retain_graph=key!=BLOCKS[-1],allow_unused=True)
                c.backs+=1
                gradients[key]=[x.detach().cpu().double() if x is not None else None for x in g]
                assert all(torch.isfinite(x).all() for x in gradients[key] if x is not None)
                norms[key]=sum(float(x.square().sum()) for x in gradients[key] if x is not None)
            dots={b:sum(float((x*y).sum()) for x,y in zip(gradients['CE'],gradients[b]) if x is not None and y is not None) for b in BLOCKS}
            rows.append(dict(dataset=c.ds,seed=c.seed,t=inp['t'],input_hash=s.input_hash([inp]),norm_squared=norms,CE_macro_inner_products=dots,
                CE=float(ce.detach()),blocks={k:float(v.detach()) for k,v in blocks.items()}))
            write(Path(out)/'calibration_rows.json',rows)
            del gradients,g,blocks,tp,pr,lo,terms,ce
        assert before==s.state_digest(c.checkpoint());calls+=c.calls;backs+=c.backs
        del c,params;torch.cuda.empty_cache()
    coeff=freeze_coefficients(rows);coeff.update(calibration_sha256=sha(Path(out)/'calibration_rows.json'),network_forwards=calls,network_VJPs=backs,seconds=time.time()-start)
    assert calls==24 and backs==96
    write(Path(out)/'gradient_coefficients.json',coeff)

def train(out,index,arm):
    c=Context(out,index);folder=Path(out)/c.row['id']/arm;folder.mkdir(parents=True,exist_ok=False)
    coeff=json.loads((Path(out)/'gradient_coefficients.json').read_text());train=c.inputs('train');valid=c.inputs('validation')
    manifest=dict(train=[dict(step=i+1,t=x['t'],hash=s.input_hash([x])) for i,x in enumerate(train)],validation=[dict(t=x['t'],hash=s.input_hash([x])) for x in valid],
        generator='indexed SeedSequence root/parent_seed/update; immutable input root and complete update count reconstruct next input')
    write(folder/'inputs.json',manifest);packed={}
    for split,ins in [('train',train),('validation',valid)]:
        for i,x in enumerate(ins):
            for k,v in x.items():packed[f'{split}_{i}__{k}']=v
    np.savez_compressed(folder/'inputs.npz',**packed)
    result=dict(dataset=c.ds,seed=c.seed,arm=arm,updates=[],evaluations={},baseline=c.evaluate(valid,coeff) if arm=='CE' else None,
        validation_overlaps_train=sorted(set(x['hash'] for x in manifest['train'])&set(x['hash'] for x in manifest['validation'])),
        coefficients_sha256=sha(Path(out)/'gradient_coefficients.json'),environment=w.common.environment('cuda'))
    write(folder/'result.json',result)
    try:
        for step,inp in enumerate(train,1):
            result['updates'].append(c.update(inp,arm,coeff,step))
            if step==1:
                st=c.checkpoint();torch.save(st,folder/'recovery_start.pt')
                c.update(train[1],arm,coeff,2);a=c.checkpoint()
                loaded=torch.load(folder/'recovery_start.pt',map_location='cpu',weights_only=False);assert s.state_digest(st)==s.state_digest(loaded)
                c.boundary=c.restore(loaded);c.update(train[1],arm,coeff,2);b=c.checkpoint()
                result['recovery']=dict(serialized_exact=True,next_update_exact={k:s.state_digest(a[k])==s.state_digest(b[k]) for k in ['model','optimizer','ema','rng','updates']},
                    model_max_abs=max(float((a['model'][k].double()-b['model'][k].double()).abs().max()) for k in a['model'] if a['model'][k].numel()),
                    note='GPU recalculation not globally certified; no tolerance/operator changes')
                assert result['recovery']['next_update_exact']['rng'] and result['recovery']['next_update_exact']['updates']
                c.boundary=c.restore(st)
            if step in (1,32,64,128):
                st=c.checkpoint();cp=folder/f'step{step}_complete.pt';torch.save(st,cp)
                loaded=torch.load(cp,map_location='cpu',weights_only=False);assert s.state_digest(st)==s.state_digest(loaded);c.boundary=c.restore(loaded)
                if step!=1:result['evaluations'][str(step)]=dict(raw=c.evaluate(valid,coeff),ema=c.evaluate(valid,coeff,True),checkpoint_sha256=sha(cp))
            write(folder/'result.json',result);write(folder/'progress.json',dict(step=step,forwards=c.calls,backwards=c.backs,epoch=time.time()))
        assert len(result['updates'])==128
        assert c.calls==(130+72+(12 if arm=='CE' else 0)) and c.backs==130
        cp=folder/'step128_complete.pt';st=c.checkpoint();effective=dict(st['model']);effective.update(st['ema']['shadow'])
        write(folder/'completion.json',dict(status='complete',updates=128,forwards=c.calls,backwards=c.backs,checkpoint=str(cp),checkpoint_sha256=sha(cp),model_hash=s.state_digest(effective),
            coefficients_sha256=sha(Path(out)/'gradient_coefficients.json'),inputs_sha256=sha(folder/'inputs.json'),peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20))
    except BaseException:
        write(folder/'failure.json',dict(traceback=traceback.format_exc(),forwards=c.calls,backwards=c.backs))
        try:torch.save(c.checkpoint(),folder/'stopped_complete.pt')
        except BaseException:write(folder/'not_resumable.json',dict(reason='boundary invalid; use previous complete checkpoint only'))
        raise

if __name__=='__main__':
    out=Path(sys.argv[1]);mode=sys.argv[2]
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('owned watchdog stop')))
    try:
        if mode=='calibrate':calibrate(out)
        else:train(out,int(sys.argv[3]),sys.argv[4])
    except BaseException:
        write(out/('worker_failure_'+mode+'_'+'_'.join(sys.argv[3:])+'.json'),dict(traceback=traceback.format_exc()));raise
