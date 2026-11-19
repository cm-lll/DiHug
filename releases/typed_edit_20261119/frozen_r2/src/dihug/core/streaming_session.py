"""PR2 side-channel logging. No proposal, model or selection calculation here.

Full state is saved initially, every 256 attempts, at the first switch,
at an explicit engineering stop, on failure and at completion. Historical
PR1 SamplingSession and its artifacts remain unchanged.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import numpy as np
from .sampling_session import SamplingSession, _jsonable, digest, save_snapshot, read_snapshot


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(_jsonable(value), f, sort_keys=True, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


class StreamingSession(SamplingSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.journal = None
        self.chain = '0'*64
        self.base_completed = 0
        self.started = time.perf_counter()
        self.failed = False

    def run(self, prior, model, role_prior, **kwargs):
        identity = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if self.restore:
            if self.restore.get('streaming_source') != identity:
                raise ValueError('streaming source identity mismatch')
            self.chain = self.restore['stream']['chain']
            self.base_completed = self.restore['completed']
        log = self.directory/'attempts.jsonl'
        if log.exists():
            if not self.restore or log.stat().st_size != self.restore['stream']['offset']:
                raise ValueError('journal/checkpoint boundary mismatch; preserve tail for explicit recovery')
        else:
            atomic_json(self.directory/'journal_base.json', {'completed':self.base_completed,'chain':self.chain})
        self.journal = log.open('a')
        try:
            result = super().run(prior, model, role_prior, **kwargs)
            return result
        finally:
            self.journal.close()

    def capture(self, graph, target_rows, values, completed):
        # records stays empty: the append-only journal owns candidate history.
        p = super().capture(graph, target_rows, values, completed)
        if self.journal:
            self.journal.flush(); os.fsync(self.journal.fileno())
        p['stream'] = {'offset':0 if self.journal is None else self.journal.tell(),
                       'chain':self.chain,'completed':completed,'base_completed':self.base_completed}
        p['streaming_source'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if self.config.get('stage1_positive_gain_guard',False):
            p['decoder_version'] = 'pr2g-positive-gain-eligibility-legacy-fallback-v1'
        self.payload = p
        return p

    def checkpoint(self, graph, target_rows, values, completed, label=None):
        p = self.capture(graph, target_rows, values, completed)
        path = self.directory/('checkpoint_%06d.json'%completed)
        save_snapshot(path,p)
        if label: save_snapshot(self.directory/(label+'.json'),p)
        atomic_json(self.directory/'progress.json', {
            'completed':completed,'checkpoint':str(path.resolve()),
            'phase':'switch' if values['switch_refinement'] else 'role',
            'error':values['current_role_error'],'applied':values['applied'],
            'elapsed_seconds':time.perf_counter()-self.started,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'status':'stopped' if self.stop_after is not None and completed>=self.stop_after else
                     ('completed' if completed==self.config['steps'] else 'running')})
        self.durable_completed = completed
        return p

    def before_attempt(self, graph, target_rows, values):
        completed = values['completed']-1
        self.diagnostics = {}
        self.before = {'state':{'current_role_error':values['current_role_error'],
                                'switch_refinement':values['switch_refinement']},'completed':completed}
        self.context = (graph,target_rows,values,completed)
        if completed == 0:
            self.checkpoint(graph,target_rows,values,0,'initial')
        if values['switch_refinement'] and getattr(self,'first_switch',None) is None:
            self.first_switch = {'k':completed+1,'a':completed,
                'remaining':self.config['steps']-completed,
                'model_time_index':self.config['model_time_start_step']+completed+1}
            self.transition_snapshot = self.checkpoint(graph,target_rows,values,completed,'transition')
            if self.stop_at_transition: return True
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 8*1024**3:
            raise MemoryError('PR2 worker peak RSS exceeded 8 GiB')
        return False

    def after_attempt(self, graph, target_rows, values):
        selected = values['selected']; candidates = values['candidates']
        def proposal(p): return {'action':p.action,'family':p.family,'old':p.old_edges,'new':p.new_edges}
        def tensor(name):
            t = values.get(name)
            return None if t is None else t.detach().cpu().tolist()
        gains = tensor('degree_gains')
        record = _jsonable({
            'attempt':values['completed'],'model_time_index':self.config['model_time_start_step']+values['completed'],
            'time':values['time_fraction'],'phase':'switch' if self.before['state']['switch_refinement'] else 'role',
            'candidates':[proposal(p) for p in candidates],'candidate_count':len(candidates),
            'legal_candidates':len(candidates),'selected_index':values['selected_index'],'selected':proposal(selected),
            'selection':self.config['selection_mode'],'decoder':self.config['decoder'],
            'noop_reason':'scorer_maximum' if selected.action=='noop' else None,
            'role_error_before':self.before['state']['current_role_error'],
            'role_error_after':float(np.square(target_rows-values['current_degrees']).sum()),
            'under_degree':np.maximum(target_rows-values['current_degrees'],0).sum(axis=0),
            'over_degree':np.maximum(values['current_degrees']-target_rows,0).sum(axis=0),
            'successful_edit':selected.action!='noop','diagnostics':copy.deepcopy(self.diagnostics),
            'structure_logits':tensor('structure_logits'),'degree_gains':gains,'scores':tensor('scores'),
            'effective_role_weight':values['effective_role_weight'],
            'guard_enabled':self.config.get('stage1_positive_gain_guard',False),
            'positive_indices':values.get('positive_indices',[]),
            'degree_gain_values':values.get('degree_gain_values'),
            'guard_applied':values.get('guard_applied',False),
            'fallback_to_legacy':values.get('fallback_to_legacy',False),
            'original_argmax':values.get('original_argmax'),
            'actual_selected':values['selected_index'],
            'improving_candidates':sum(g>0 for g in gains),
            'selected_improving':gains[values['selected_index']]>0,
            'cumulative_proposed':values['proposed'],'cumulative_applied':values['applied'],
            'action_counts':copy.deepcopy(values['action_counts']),
            'transition_step':values['transition_step'],'transition_reason':values['transition_reason'],
            'proposal_rng_after':values['proposal_rng'].bit_generator.state,
            'selection_rng_after':values['selection_rng'].bit_generator.state})
        if record['role_error_after'] != values['current_role_error']:
            raise RuntimeError('stream degree residual differs from sampler incremental residual')
        self.chain = digest({'previous':self.chain,'record':record})
        self.journal.write(json.dumps({'chain':self.chain,'elapsed_seconds':time.perf_counter()-self.started,
                                      'record':record},separators=(',',':'),allow_nan=False)+'\n')
        self.journal.flush()
        k = values['completed']
        self.context = (graph,target_rows,values,k)
        if k%1000 == 0:
            atomic_json(self.directory/('window_state_%06d.json'%k),{
                'attempt':k,'error':record['role_error_after'],'under_degree':record['under_degree'],
                'over_degree':record['over_degree'],'counts':{f:len(e) for f,e in graph.family_edges.items()},
                'target_feasibility':self.target_feasibility,'phase':record['phase'],
                'transition_step':values['transition_step']})
        stopping = self.stop_after is not None and k>=int(self.stop_after)
        if k%256 == 0 or k==128 or k==self.config['steps'] or stopping:
            self.checkpoint(graph,target_rows,values,k)
        return stopping

    def failure(self, exc):
        if self.failed: return
        self.failed = True
        graph,target,values,completed = self.context
        try:
            p = self.capture(graph,target,values,completed)
            p['failure'] = {'type':type(exc).__name__,'message':str(exc),
                'failed_attempt':completed+1,'diagnostics':self.diagnostics,
                'classification':'CANDIDATE_COVERAGE_UNKNOWN; exhaustion does not prove no legal edit',
                'last_durable_completed':getattr(self,'durable_completed',self.base_completed),
                'state_trust':'verify graph/error/RNG before any continuation'}
            save_snapshot(self.directory/'failure.json',p)
        finally:
            atomic_json(self.directory/'failure_status.json',{
                'status':'failed','completed':completed,'type':type(exc).__name__,'message':str(exc)})

    @classmethod
    def load(cls,path,directory,*,stop_after=None):
        p = read_snapshot(path); x = p['experiment']
        s = cls(directory,**x,target_rows=p['target_rows'],stop_after=stop_after)
        s.restore=p; s.first_switch=p['first_switch']; return s
