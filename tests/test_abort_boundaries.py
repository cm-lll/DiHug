"""Small CPU fault tests using the ACTUAL update and sampling entry functions.

Only the denoiser/context is a tiny synthetic test double. The query planner,
forward noise, CE, AdamW, EMA, reverse kernel, sampler contracts and serializers
are the real implementation. No scientific model/checkpoint/data is loaded.
"""
import copy
import hashlib
import importlib.util
import json
import os
import pathlib
import random
import sys
import tempfile
import subprocess
import types
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCI = ROOT / 'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['DIHUG_O_CN_RUN'] = str(ROOT / 'results/o_cn_matched_formal_test')
sys.path[:0] = [str(ROOT / 'experiments/o_cn_matched_execute_v1'), str(SCI / 'package'), str(SCI)]
import numpy as np
import torch
from boundary_transactions import TrainingBoundary, SamplingBoundary, rng_capture, rng_restore
from ema import ParameterEMA
from integration import restore_payload, checkpoint_payload
from probe import input_hash, equal
from hsgd.bw_preflight import common
from hsgd import runtime
from observers import RecordedGenerator, RecordedPlan, digest

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy_path(relative):
    # Obtain the EXACT reviewed Git blob, not an editable sibling directory.
    path=TEMP/'reviewed_reference'/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(subprocess.check_output(['git','show','2f2cb300c1c7efc3dd6a28d462d5cb7584e88919:'+relative],cwd=ROOT))
    return path


def stable(value):
    h = hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, np.ndarray):
            h.update(v.tobytes())
        elif isinstance(v, dict):
            for k in sorted(v):
                h.update(str(k).encode()); add(v[k])
        elif isinstance(v, (list, tuple)):
            for x in v: add(x)
        else:
            h.update(repr(v).encode())
    add(value)
    return h.hexdigest()


def reset(seed=7021):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return np.random.default_rng(seed + 8)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.lin_out_E = torch.nn.Linear(3, 2)
        self.register_buffer('prior', torch.tensor(common.toy_case()['prior'],dtype=torch.float64))
        self.register_buffer('forward_counter', torch.zeros((), dtype=torch.long))
        self.drop = torch.nn.Dropout(.2)


class TinyContext:
    fail_forward = False
    def __init__(self, model, types_, edges, families, token):
        self.model = model
    def predict(self, query, family, time, token, details):
        # Exercise buffer, global Python/NumPy and Torch RNG rollback, too.
        self.model.forward_counter.add_(1)
        jitter = random.random() + float(np.random.random())
        x = torch.tensor(np.c_[query.T / 6., np.full(len(family), time)], dtype=torch.float32)
        logits = self.model.backbone.lin_out_E(self.model.drop(x + jitter / 100))
        if self.fail_forward:
            raise RuntimeError('injected forward failure')
        return types.SimpleNamespace(query_families=torch.tensor(family)), logits, logits.softmax(-1)[:, 1]
    def close(self):
        pass


class TrainingTests(unittest.TestCase):
    def run_path(self, runtime_name, fault=None, old=False):
        relative='experiments/'+runtime_name+'/single_input.py'
        module = load(legacy_path(relative) if old else ROOT/relative, 'update_' + runtime_name + str(old))
        module.FrozenTypedContext = TinyContext
        rng = reset(); model = TinyModel(); opt = torch.optim.AdamW(model.parameters(), lr=.0002, weight_decay=1e-12)
        ema = ParameterEMA(model); tx = TrainingBoundary(model, opt, ema, rng, 'cpu', 0)
        case = common.toy_case(); cfg = dict(T=1000, seed=100)
        binding = dict(test='synthetic', runtime=runtime_name)
        trace = []
        for position in range(1, 5):
            if fault is not None and position == 2:
                before = tx.checkpoint(binding)
                with self.assertRaises((RuntimeError, FloatingPointError, ValueError)):
                    with tx.step(position):
                        inp = module.one_input(case, cfg, position, rng)
                        if fault == 'input':
                            raise ValueError('injected input contract failure')
                        if fault == 'forward':
                            TinyContext.fail_forward = True
                        handle = None
                        if fault == 'gradient':
                            handle = next(model.parameters()).register_hook(lambda g: g * float('nan'))
                        try:
                            module.update(model,opt,case,inp,1000,'cpu',ema,tx.completed,transaction=tx)
                        finally:
                            TinyContext.fail_forward = False
                            if handle: handle.remove()
                self.assertTrue(tx.resumable)
                self.assertEqual(tx.completed, 1)
                self.assertTrue(equal(before, tx.checkpoint(binding)))
                # Round-trip through a real CPU checkpoint and NEW objects.
                with tempfile.TemporaryDirectory(dir=TEMP) as d:
                    p = pathlib.Path(d) / 'boundary.pt'; torch.save(tx.checkpoint(binding), p)
                    payload = torch.load(p, map_location='cpu')
                model = TinyModel(); opt = torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=1e-12)
                ema = ParameterEMA(model); rng = np.random.default_rng(999)
                completed = restore_payload(payload,model,opt,ema,binding,lambda z: rng_restore(z,rng,'cpu'))
                tx = TrainingBoundary(model,opt,ema,rng,'cpu',completed)
            if old:
                inp = module.one_input(case,cfg,position,rng)
                module.update(model,opt,case,inp,1000,'cpu',ema,position-1)
                payload = checkpoint_payload(model,opt,ema,position,rng_capture(rng,'cpu'),binding)
            else:
                with tx.step(position):
                    inp = module.one_input(case,cfg,position,rng)
                    module.update(model,opt,case,inp,1000,'cpu',ema,tx.completed,transaction=tx)
                payload = tx.checkpoint(binding)
            # Do not compare extra recovery metadata or timing fields.
            trace.append((input_hash([inp]), stable({k:payload[k] for k in ['model','optimizer','ema','rng','updates']}),
                          stable({k:p.grad for k,p in model.named_parameters()})))
        return trace

    def test_normal_path_and_three_failure_sites_both_actual_updates(self):
        for name in ('o_cn_matched_execute_v1', 'late_final_single_input_v1'):
            with self.subTest(runtime=name):
                expected = self.run_path(name,old=True)
                self.assertEqual(expected,self.run_path(name))
                for fault in ('input','forward','gradient'):
                    with self.subTest(fault=fault):
                        self.assertEqual(expected,self.run_path(name,fault=fault))

    def test_optimizer_partial_write_is_not_resumable(self):
        module = load(ROOT/'experiments/late_final_single_input_v1/single_input.py','dirty_update')
        module.FrozenTypedContext = TinyContext
        rng=reset();model=TinyModel();opt=torch.optim.AdamW(model.parameters(),lr=.0002);ema=ParameterEMA(model)
        tx=TrainingBoundary(model,opt,ema,rng,'cpu',0);old=tx.checkpoint({})
        def broken_step():
            with torch.no_grad():next(model.parameters()).add_(1.)
            raise RuntimeError('injected partial optimizer write')
        with mock.patch.object(opt,'step',broken_step), self.assertRaises(RuntimeError):
            with tx.step(1):
                inp=module.one_input(common.toy_case(),dict(T=1000,seed=1),1,rng)
                module.update(model,opt,common.toy_case(),inp,1000,'cpu',ema,0,transaction=tx)
        self.assertFalse(tx.resumable);self.assertFalse(ema.at_update_boundary)
        with self.assertRaises(RuntimeError):tx.checkpoint({})
        self.assertEqual(old['updates'],0)  # the PREVIOUS independent checkpoint remains valid

    def test_partial_ema_and_failed_rng_restore_are_not_resumable(self):
        module=load(ROOT/'experiments/o_cn_matched_execute_v1/single_input.py','dirty_ema')
        module.FrozenTypedContext=TinyContext
        for failure in ('ema','rng_restore'):
            rng=reset();model=TinyModel();opt=torch.optim.AdamW(model.parameters(),lr=.0002);ema=ParameterEMA(model)
            tx=TrainingBoundary(model,opt,ema,rng,'cpu',0)
            def broken_ema(*args):
                next(iter(ema.shadow.values())).add_(1.)
                raise RuntimeError('injected partial EMA write')
            context=mock.patch.object(ema,'update',broken_ema) if failure=='ema' else mock.patch('boundary_transactions.rng_restore',side_effect=RuntimeError('unavailable backend'))
            with context, self.assertRaises(RuntimeError):
                with tx.step(1):
                    inp=module.one_input(common.toy_case(),dict(T=1000,seed=1),1,rng)
                    if failure=='rng_restore':raise RuntimeError('pre-optimizer failure')
                    module.update(model,opt,common.toy_case(),inp,1000,'cpu',ema,0,transaction=tx)
            self.assertFalse(tx.resumable)
            with self.assertRaises(RuntimeError):tx.checkpoint({})

    def test_all_three_worker_loops_wrap_input_before_update(self):
        import ast
        for relative in ('experiments/o_cn_matched_execute_v1/worker_run.py',
                         'experiments/late_final_single_input_v1/worker_run.py',
                         'experiments/d_late_ard_runtime_v1/generation_worker.py'):
            tree=ast.parse((ROOT/relative).read_text());found=[]
            for node in ast.walk(tree):
                if isinstance(node,ast.With) and any(isinstance(v.context_expr,ast.Call) and isinstance(v.context_expr.func,ast.Attribute) and v.context_expr.func.attr=='step' for v in node.items):
                    calls=[n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
                    self.assertTrue(any(c.func.id=='one_input' for c in calls))
                    update=next(c for c in calls if c.func.id=='update')
                    self.assertTrue(any(k.arg=='transaction' for k in update.keywords));found.append(node)
            self.assertEqual(len(found),1,relative)

    def test_legacy_input_failure_would_save_advanced_rng_with_old_weights(self):
        module=load(legacy_path('experiments/o_cn_matched_execute_v1/single_input.py'),'legacy_input_failure')
        rng=reset();model=TinyModel();opt=torch.optim.AdamW(model.parameters(),lr=.0002);ema=ParameterEMA(model)
        before=checkpoint_payload(model,opt,ema,0,rng_capture(rng,'cpu'),{})
        module.one_input(common.toy_case(),dict(T=1000,seed=1),1,rng)
        old_stop=checkpoint_payload(model,opt,ema,0,rng_capture(rng,'cpu'),{})
        self.assertTrue(equal(before['model'],old_stop['model']))
        self.assertEqual(old_stop['updates'],0)
        self.assertFalse(equal(before['rng'],old_stop['rng']))
        with self.assertRaises(ValueError):restore_payload(old_stop,model,opt,ema,{},lambda z:rng_restore(z,rng,'cpu'))

    def test_committed_counter_survives_exception_before_worker_assignment(self):
        module=load(ROOT/'experiments/o_cn_matched_execute_v1/single_input.py','commit_counter')
        module.FrozenTypedContext=TinyContext
        rng=reset();model=TinyModel();opt=torch.optim.AdamW(model.parameters(),lr=.0002);ema=ParameterEMA(model)
        tx=TrainingBoundary(model,opt,ema,rng,'cpu',0)
        with self.assertRaises(RuntimeError):
            with tx.step(1):
                inp=module.one_input(common.toy_case(),dict(T=1000,seed=1),1,rng)
                module.update(model,opt,common.toy_case(),inp,1000,'cpu',ema,0,transaction=tx)
            raise RuntimeError('log/worker counter assignment failed after successful commit')
        saved=tx.checkpoint({})
        self.assertEqual(saved['updates'],1);self.assertEqual(saved['ema']['updates'],1)
        self.assertTrue(equal(saved['rng'],rng_capture(rng,'cpu')))

    def test_unexpected_parameter_or_buffer_replacement_fails_closed(self):
        for failure in ('parameter_write','buffer_replacement'):
            rng=reset();model=TinyModel();opt=torch.optim.AdamW(model.parameters(),lr=.0002);ema=ParameterEMA(model)
            tx=TrainingBoundary(model,opt,ema,rng,'cpu',0)
            with self.assertRaises(RuntimeError):
                with tx.step(1):
                    rng.random(3)
                    if failure=='parameter_write':
                        with torch.no_grad():next(model.parameters()).add_(1.)
                    else:model.forward_counter=model.forward_counter.clone()
                    raise RuntimeError('unexpected mutation before optimizer')
            self.assertFalse(tx.resumable)
            with self.assertRaises(RuntimeError):tx.checkpoint({})


class TinySampler(torch.nn.Module):
    def __init__(self):
        super().__init__();self.joint_head=torch.nn.Sequential(torch.nn.Linear(739,128))
    def cuda(self):
        return self  # synthetic CPU backend ONLY in tests


class TinyPredictor:
    fail_t = None
    def __init__(self,model,case,state,kernel,t,backend,event,check_time,profile=False):
        self.blocks=0;self.times={};self.t=t
    def __call__(self,frozen,ids,fam,query):
        self.blocks+=1
        jitter=(random.random()+float(np.random.random())+float(torch.rand(())))/20
        if self.t==self.fail_t:raise RuntimeError('failure inside reverse after uniform and prediction RNG')
        return np.full(len(ids),.2+jitter)
    def close(self):
        pass


class SamplingTests(unittest.TestCase):
    def make_environment(self, folder):
        from worker import atomic_checkpoint
        reset();model=TinySampler();cfg=dict(T=1000,datasets={'toy':{'type_names':['x','y']}})
        case=common.toy_case();ss=2026102821
        binding=dict(train_seed=1,dataset='toy',config=cfg)
        path=folder/'readout.pt'
        atomic_checkpoint(path,dict(model=model.state_dict(),binding=binding,train_seed=1,updates=3000,
          readout='ema_0.99',decay=.99,checkpoint_sha256='test-final'))
        (folder/'final.sha256.json').write_text(json.dumps(dict(sha256='test-final')))
        row=dict(readout_file=str(path),train_seed=1,configuration='O_late',dataset='toy',id='synthetic')
        rng=RecordedGenerator(ss,case['space'].total);state=(rng.random(case['space'].total)<np.repeat(case['prior'],case['space'].counts)).astype(np.uint8)
        contract=dict(initial_state_sha256=digest(state),initial_uniform_sha256=rng.last_sha,steps=[])
        for t in range(1000,0,-1):
            p=RecordedPlan(common.plan_for(case,[ss,t,61],1));p.block(0);rng.random(case['space'].total)
            contract['steps'].append(dict(t=t,query_plan_sha256=p.h.hexdigest(),uniform_sha256=rng.last_sha))
        (folder/'prepared').mkdir();(folder/'prepared/sampling_contracts.json').write_text(json.dumps({'toy_'+str(ss):contract}))
        return case,cfg,row,ss

    def run_sampler(self,module,root,case,cfg,row,ss,name,fault=None,resume=None):
        reset(442);trace=[];original=runtime.reverse_step
        # Production arguments are retained; only the tiny test backend is CPU.
        def reverse(model,case_,state,kernel,t,rounds,seed,rng,device,**kw):
            candidate,rec=original(model,case_,state,kernel,t,rounds,seed,rng,'cpu',**kw)
            if fault=='coverage' and t==983:rec['exact_coverage']=False
            if fault=='contract' and t==983:rng.last_sha='injected wrong contract'
            if fault=='inside' and t==983:raise RuntimeError('after reverse consumes RNG, before caller commit')
            return candidate,rec
        def make_model(*args):return TinySampler()
        def record_commit(this,step,validate):
            result=advance(this,step,validate)
            trace.append((this.next_t,digest(this.state),stable(this.checkpoint({})['rng'])))
            return result
        advance=SamplingBoundary.advance
        output=root/name
        with mock.patch.object(module,'make_model',make_model), mock.patch.object(module,'evaluate',lambda *a:{}), mock.patch.object(module,'reverse_step',reverse), mock.patch.object(runtime,'StepPredictor',TinyPredictor), mock.patch.object(TinyPredictor,'fail_t',983 if fault=='backend' else None), mock.patch.object(torch.cuda,'empty_cache',lambda:None), mock.patch.object(SamplingBoundary,'advance',record_commit):
            if fault:
                with self.assertRaises((RuntimeError,ValueError)):
                    module.sample_one(row,ss,output,case,cfg,None,root,lambda:None,lambda **k:None)
            else:
                module.sample_one(row,ss,output,case,cfg,None,root,lambda:None,lambda **k:None,resume_chain=resume)
        return output,trace

    def test_actual_O_sampling_outer_checks_and_resume(self):
        module=load(ROOT/'experiments/o_cn_matched_execute_v1/sampling.py','actual_sampler')
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=pathlib.Path(d);case,cfg,row,ss=self.make_environment(root)
            control,trace=self.run_sampler(module,root,case,cfg,row,ss,'control')
            for fault in ('coverage','contract','inside','backend'):
                interrupted,prefix=self.run_sampler(module,root,case,cfg,row,ss,fault,fault=fault)
                saved=torch.load(interrupted/'stopped_chain.pt',map_location='cpu')
                self.assertEqual(saved['next_t'],983);self.assertEqual(saved['draw_calls'],18)
                self.assertEqual(prefix,trace[:17])
                self.assertEqual(stable(saved['rng']),trace[16][2])
                resumed,suffix=self.run_sampler(module,root,case,cfg,row,ss,fault+'_resumed',resume=interrupted/'stopped_chain.pt')
                self.assertEqual(prefix+suffix,trace)
                a=np.load(control/'graph.npz');b=np.load(resumed/'graph.npz')
                for k in a.files:self.assertTrue(np.array_equal(a[k],b[k]))

    def test_AE_sampling_uses_same_transaction_and_O_normal_matches_legacy(self):
        # Actual A/E shares this entire sampling body; only readout shape auth differs.
        a=(ROOT/'experiments/late_final_single_input_v1/sampling.py').read_text().split('def sample_one',1)[1]
        b=(ROOT/'experiments/o_cn_matched_execute_v1/sampling.py').read_text().split('def sample_one',1)[1]
        self.assertEqual(a,b)
        fixed=load(ROOT/'experiments/o_cn_matched_execute_v1/sampling.py','normal_fixed')
        legacy=load(legacy_path('experiments/o_cn_matched_execute_v1/sampling.py'),'normal_legacy')
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            root=pathlib.Path(d);case,cfg,row,ss=self.make_environment(root)
            control,trace=self.run_sampler(fixed,root,case,cfg,row,ss,'fixed')
            # Legacy has no resume argument and reads RNG on hard-coded cuda.
            from probe import rng_get
            legacy.rng_get=lambda r,d:rng_get(r,'cpu')
            original=runtime.reverse_step
            legacy_trace=[]
            def reverse(m,c,s,k,t,R,seed,rng,dev,**kw):
                value,record=original(m,c,s,k,t,R,seed,rng,'cpu',**kw)
                legacy_trace.append((t-1,digest(value),stable(rng_capture(rng,'cpu'))))
                return value,record
            reset(442)
            with mock.patch.object(legacy,'make_model',lambda *a:TinySampler()), mock.patch.object(legacy,'evaluate',lambda *a:{}), mock.patch.object(legacy,'reverse_step',reverse), mock.patch.object(runtime,'StepPredictor',TinyPredictor), mock.patch.object(torch.cuda,'empty_cache',lambda:None):
                legacy.sample_one(row,ss,root/'legacy',case,cfg,None,root,lambda:None,lambda **k:None)
            self.assertEqual(trace,legacy_trace)
            x=np.load(control/'graph.npz');y=np.load(root/'legacy/graph.npz')
            for k in x.files:self.assertTrue(np.array_equal(x[k],y[k]))
            f=torch.load(control/'complete_t0000.pt',map_location='cpu');g=torch.load(root/'legacy/complete_t0000.pt',map_location='cpu')
            self.assertEqual(f['next_t'],g['next_t']);self.assertEqual(f['draw_calls'],g['draw_calls'])
            for k,v in g['rng'].items():self.assertTrue(equal(v,f['rng'][k]))
            # Reproduce the OLD bug: calls look correct but true RNG is one draw ahead.
            prior={}
            def interrupted(m,c,s,k,t,R,seed,rng,dev,**kw):
                value,record=original(m,c,s,k,t,R,seed,rng,'cpu',**kw)
                if t==984:prior.update(rng_capture(rng,'cpu'))
                if t==983:record['exact_coverage']=False
                return value,record
            reset(442)
            with mock.patch.object(legacy,'make_model',lambda *a:TinySampler()), mock.patch.object(legacy,'evaluate',lambda *a:{}), mock.patch.object(legacy,'reverse_step',interrupted), mock.patch.object(runtime,'StepPredictor',TinyPredictor), mock.patch.object(torch.cuda,'empty_cache',lambda:None):
                with self.assertRaises(ValueError):legacy.sample_one(row,ss,root/'legacy_failure',case,cfg,None,root,lambda:None,lambda **k:None)
            mixed=torch.load(root/'legacy_failure/stopped_chain.pt',map_location='cpu')
            self.assertEqual(mixed['next_t'],983);self.assertEqual(mixed['draw_calls'],18)
            self.assertFalse(equal(mixed['rng']['numpy'],prior['numpy']))

    def test_legacy_mixed_checkpoint_and_rng_restore_failure_rejected(self):
        rng=RecordedGenerator(7,6);rng.random(6);tx=SamplingBoundary(np.zeros(6,dtype=np.uint8),5,rng,'cpu')
        old=tx.checkpoint({});old.pop('recovery')
        with self.assertRaises(ValueError):tx.restore(old,{})
        with mock.patch('boundary_transactions.rng_restore',side_effect=RuntimeError('backend lost')):
            with self.assertRaises(ValueError):
                tx.advance(lambda s,t:(rng.random(6),{}),lambda s,r:(_ for _ in ()).throw(ValueError('contract')))
        self.assertFalse(tx.resumable)
        with self.assertRaises(RuntimeError):tx.checkpoint({})


TEMP = pathlib.Path(os.environ.get('DIHUG_TEST_TMP',str(ROOT.parent.parent/'results/abort_boundary_audit_20261104/temporary')))
TEMP.mkdir(parents=True,exist_ok=True)
if __name__=='__main__':unittest.main(verbosity=2)
