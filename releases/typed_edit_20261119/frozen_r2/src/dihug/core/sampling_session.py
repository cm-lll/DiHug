"""Opt-in CPU experiment state, counter streams and atomic recovery.

Legacy CLI is unchanged. This is a sampler interface, not a real-data runner.
Only scalar motif summaries (never reference adjacency) reach the rule arm.
"""
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from .graph import TypedGraph, FamilySpec

STATE_NAMES = ('effective_role_weight', 'current_role_error', 'relative_role_error_target',
    'switch_refinement', 'transition_step', 'transition_reason', 'plateau_reference_step',
    'plateau_reference_error', 'plateau_noops', 'plateau_observations', 'trace',
    'action_counts', 'proposed', 'applied', 'time_horizon')


def _jsonable(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, torch.device): return str(x)
    if isinstance(x, dict): return {k:_jsonable(v) for k,v in x.items()}
    if isinstance(x, (tuple, list)): return [_jsonable(v) for v in x]
    return x


def digest(payload):
    return hashlib.sha256(json.dumps(_jsonable(payload), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def counter_rng(dataset, seed, repeat, attempt, stream, arm=None):
    # Explicitly length-delimited UTF-8 fields; proposal identity is arm-free.
    fields = ['PR1/20261007', dataset, seed, repeat, attempt, stream]
    if stream == 'selection': fields.append(arm)
    elif stream != 'proposal': raise ValueError('unknown random stream')
    h = hashlib.sha256()
    for value in fields:
        b = str(value).encode(); h.update(len(b).to_bytes(8, 'big')); h.update(b)
    return np.random.Generator(np.random.PCG64(int.from_bytes(h.digest()[:16], 'big')))


def model_identity(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        a = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(tuple(a.shape)).encode())
        h.update(a.numpy().tobytes())
    return {'class': type(model).__module__+'.'+type(model).__qualname__, 'sha256':h.hexdigest(),
            'input_dim':getattr(model, 'input_dim', None), 'dtype':str(next(model.parameters()).dtype)}


def source_identity():
    root = Path(__file__).parent
    return {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in
            ['graph.py','mixed_edit_diffusion.py','role_guided_edit_diffusion.py','sampling_session.py',
             'target_feasibility.py','attribution_motifs.py']}


def graph_payload(graph):
    return {'node_types':graph.node_types.tolist(), 'type_names':list(graph.type_names),
            'families':[{'name':name,'src_type':s.src_type,'dst_type':s.dst_type,
                         'src_nodes':list(s.src_nodes),'dst_nodes':list(s.dst_nodes),
                         'edges':[list(e) for e in graph.family_edges[name]]}
                        for name,s in graph.family_specs.items()]}


def restore_graph(payload):
    specs = {f['name']:FamilySpec(f['name'], f['src_type'], f['dst_type'],
                               tuple(f['src_nodes']),tuple(f['dst_nodes'])) for f in payload['families']}
    graph = TypedGraph(np.array(payload['node_types']), payload['type_names'], specs,
                       {f['name']:[tuple(e) for e in f['edges']] for f in payload['families']})
    graph.validate(); graph.validate_disjoint_family_domains(); return graph


def global_rng_state():
    ns = np.random.get_state()
    return {'python':_jsonable(random.getstate()), 'numpy':_jsonable(ns),
            'torch_cpu':torch.get_rng_state().tolist(),
            'torch_cuda': [s.tolist() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else [],
            'cuda_initialized':torch.cuda.is_initialized()}


def restore_global_rng(payload):
    def tuples(x): return tuple(tuples(v) for v in x) if isinstance(x,list) else x
    random.setstate(tuples(payload['python']))
    ns = payload['numpy']; np.random.set_state((ns[0],np.array(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]))
    torch.set_rng_state(torch.tensor(payload['torch_cpu'],dtype=torch.uint8))
    if payload['cuda_initialized']:
        if not torch.cuda.is_initialized(): raise ValueError('CUDA RNG context missing; CPU session cannot initialize it')
        torch.cuda.set_rng_state_all([torch.tensor(s,dtype=torch.uint8) for s in payload['torch_cuda']])


def save_snapshot(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = {'format':'dihug-pr1-cpu-snapshot-v1', 'sha256':digest(payload), 'payload':_jsonable(payload)}
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w') as f:
        json.dump(wrapper, f, sort_keys=True, allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def read_snapshot(path):
    w = json.loads(Path(path).read_text())
    if w.get('format') != 'dihug-pr1-cpu-snapshot-v1' or w.get('sha256') != digest(w.get('payload')):
        raise ValueError('snapshot integrity/version mismatch')
    p = w['payload']
    if p.get('target_hash') != digest(p.get('target_rows')):
        raise ValueError('snapshot target identity mismatch')
    if not 0 <= p['completed'] <= p['config']['steps']:
        raise ValueError('snapshot progress outside budget')
    return p


class SamplingSession:
    def __init__(self, directory, *, rng_mode='legacy', dataset='toy', repeat=0,
                 arm='M', target_rows=None, templates=None, reference_counts=None,
                 stop_after=None, stop_at_transition=False):
        if rng_mode not in ('legacy','counter') or arm not in ('M','R','C'):
            raise ValueError('unsupported session random stream/arm')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rng_mode, self.dataset, self.repeat, self.arm = rng_mode, dataset, int(repeat), arm
        self.target_rows = None if target_rows is None else np.array(target_rows,copy=True)
        self.templates = templates
        self.reference_counts = reference_counts
        self.stop_after, self.stop_at_transition = stop_after, stop_at_transition
        self.restore = None; self.records = []; self.transition_snapshot = None
        self.diagnostics = {}; self.payload = None; self.config = None

    def run(self, prior, model, role_prior, **kwargs):
        from .role_guided_edit_diffusion import sample_role_guided_edit_diffusion
        sig = inspect.signature(sample_role_guided_edit_diffusion)
        b = sig.bind(prior, model, role_prior, **kwargs); b.apply_defaults()
        self.config = _jsonable({k:v for k,v in b.arguments.items() if k not in
                               ('prior','model','role_prior','initial_graph','runtime')})
        if self.config['sampling_backend'] != 'cpu' or str(self.config['device']) != 'cpu':
            raise ValueError('experiment session requires CPU backend/device')
        if self.arm == 'C' and self.config['phase_policy'] != 'switch_only':
            raise ValueError('rule arm is only defined for switch_only')
        if self.arm == 'C' and (self.templates is None or self.reference_counts is None):
            raise ValueError('rule arm needs frozen templates/public scalar counts')
        self.identities = {'model':model_identity(model), 'prior':digest(prior.to_json_dict()),
                           'role_prior':digest(role_prior.to_json_dict()), 'source':source_identity()}
        if self.restore:
            if self.identities != self.restore['identities'] or self.config != self.restore['config']:
                raise ValueError('snapshot model/source/prior/config mismatch')
            self.records = copy.deepcopy(self.restore['records'])
            self.target_rows = np.array(self.restore['target_rows'])
            restore_global_rng(self.restore['global_rng'])
        try:
            return sample_role_guided_edit_diffusion(prior, model, role_prior, runtime=self, **kwargs)
        except Exception as exc:
            if hasattr(self,'before'): self.failure(exc)
            raise

    def streams(self, legacy_rng, seed, attempt):
        if self.rng_mode == 'legacy': return legacy_rng, legacy_rng
        return (counter_rng(self.dataset,seed,self.repeat,attempt,'proposal'),
                counter_rng(self.dataset,seed,self.repeat,attempt,'selection',self.arm))

    def capture(self, graph, target_rows, values, completed):
        self.payload = {'config':self.config, 'identities':self.identities, 'graph':graph_payload(graph),
            'target_rows':_jsonable(target_rows), 'target_hash':digest(target_rows),
            'state':_jsonable({n:values[n] for n in STATE_NAMES}), 'completed':int(completed),
            'numpy_generator':copy.deepcopy(values['rng'].bit_generator.state), 'global_rng':global_rng_state(),
            'records':copy.deepcopy(self.records), 'experiment':{'rng_mode':self.rng_mode,'dataset':self.dataset,
            'repeat':self.repeat,'arm':self.arm,'templates':_jsonable(self.templates),'reference_counts':self.reference_counts},
            'first_switch':getattr(self,'first_switch',None), 'target_feasibility':self.target_feasibility}
        return copy.deepcopy(self.payload)

    def before_attempt(self, graph, target_rows, values):
        completed = values['completed']-1
        self.diagnostics = {}
        self.before = self.capture(graph,target_rows,values,completed)
        if values['switch_refinement'] and getattr(self,'first_switch',None) is None:
            self.first_switch = {'k':completed+1, 'a':completed,
                                 'remaining':self.config['steps']-completed,
                                 'model_time_index':self.config['model_time_start_step']+completed+1}
            self.before['first_switch'] = self.first_switch
            self.transition_snapshot = copy.deepcopy(self.before)
            save_snapshot(self.directory/'transition.json',self.before)
            if self.stop_at_transition:
                save_snapshot(self.directory/'progress.json',self.before); return True
        return False

    def after_attempt(self, graph, target_rows, values):
        from .role_guided_edit_diffusion import relation_degree_error
        selected = values['selected']; scores = values.get('scores')
        candidates = values['candidates']
        before_error = self.before['state']['current_role_error']
        actual_error = relation_degree_error(graph,target_rows,tuple(sorted(graph.family_specs)))
        def proposal(p): return {'action':p.action,'family':p.family,'old':p.old_edges,'new':p.new_edges}
        self.records.append(_jsonable({'attempt':values['completed'],
            'model_time_index':self.config['model_time_start_step']+values['completed'],
            'time':values['time_fraction'], 'phase':'switch' if self.before['state']['switch_refinement'] else 'role',
            'candidates':[proposal(p) for p in candidates], 'candidate_count':len(candidates),
            'legal_candidates':len(candidates), 'selected_index':values['selected_index'],
            'selected':proposal(selected), 'selection':self.arm if self.config['phase_policy']=='switch_only' else self.config['selection_mode'],
            'noop_reason':('random_selected' if self.arm=='R' or self.config['selection_mode']=='random' else
                           'scorer_or_rule_maximum') if selected.action=='noop' else None,
            'role_error_before':before_error,'role_error_after':actual_error,
            'under_degree':np.maximum(target_rows-values['current_degrees'],0).sum(axis=0),
            'over_degree':np.maximum(values['current_degrees']-target_rows,0).sum(axis=0),
            'successful_edit':selected.action!='noop','diagnostics':copy.deepcopy(self.diagnostics),
            'scores':None if scores is None else scores.detach().cpu().tolist(),
            'proposal_rng_after':values['proposal_rng'].bit_generator.state,
            'selection_rng_after':values['selection_rng'].bit_generator.state}))
        p = self.capture(graph,target_rows,values,values['completed'])
        save_snapshot(self.directory/'progress.json',p)
        return self.stop_after is not None and values['completed']>=int(self.stop_after)

    def failure(self, exc):
        p = copy.deepcopy(self.before)
        no_switch = p['state']['switch_refinement'] and all(len(f['edges']) < 2 for f in p['graph']['families'])
        p['failure'] = {'type':type(exc).__name__,'message':str(exc),
                        'failed_attempt':p['completed']+1,'diagnostics':self.diagnostics,
                        'classification':('PROVED_NO_SWITCH_EACH_FAMILY_HAS_FEWER_THAN_TWO_EDGES' if no_switch else
                                          'CANDIDATE_COVERAGE_UNKNOWN; exhaustion does not prove no legal edit')}
        save_snapshot(self.directory/'failure.json',p)

    @classmethod
    def load(cls, path, directory, *, stop_after=None):
        p = read_snapshot(path); x = p['experiment']
        s = cls(directory, **x, target_rows=p['target_rows'],stop_after=stop_after)
        s.restore = p; s.first_switch = p['first_switch']; return s

    def resume(self, prior, model, role_prior):
        if not self.restore: raise ValueError('no restore payload')
        return self.run(prior,model,role_prior,**self.restore['config'])

    @classmethod
    def fork(cls, transition_path, directory, *, arm, dataset, repeat, templates, reference_counts):
        s = cls.load(transition_path,directory)
        if s.restore['first_switch'] is None: raise ValueError('not a transition snapshot')
        s.rng_mode, s.arm, s.dataset, s.repeat = 'counter',arm,dataset,int(repeat)
        s.templates,s.reference_counts = templates,reference_counts
        s.restore['config']['phase_policy'] = 'switch_only'
        s.restore['config']['selection_mode'] = 'model'
        s.restore['config']['fixed_role_steps'] = None
        s.restore['state']['switch_refinement'] = True
        return s
