"""Bind the full four-cell panel before any new optimizer/model calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,hashlib,datetime,zipfile
from pathlib import Path
from scoring import *
def main():
    out=Path(sys.argv[1]).resolve();out.mkdir(exist_ok=False);(out/'logs').mkdir()
    old=json.loads((ROOT/'results/paper_ablation_20261112T190000/protocol.json').read_text())
    gate=ROOT/'results/joint_transport_score_gate_20261113';assert json.loads((gate/'receipt.json').read_text())['status']=='passed'
    start=datetime.datetime.fromisoformat('2026-11-13T01:19:28+08:00').timestamp()
    bindings={}
    def bind(path,expected=None):
        path=str(Path(path).resolve());h=base.sha(path)
        if expected is not None:assert h==expected,path
        bindings[path]=h;return path
    parents=old['parents'];reused=old['reused_graphs']
    for row in parents.values():
        bind(row['data'],old['bindings'][row['data']]);bind(row['readout'],old['bindings'][row['readout']])
        z=torch.load(row['readout'],map_location='cpu',weights_only=False)
        assert z['updates']==3000 and z['train_seed']==row['train_seed'] and z['readout']=='ema_0.99'
        cfg=z['binding']['config'];assert cfg['inputs_per_update']==1 and cfg['T']==1000 and cfg['rounds']==1
        assert cfg==row['config'];del z
    for item in reused:
        bind(item['path'],old['bindings'][item['path']]);folder=Path(item['path']).parent
        for name in ('identity.json','summary.json','steps.jsonl','binding.json'):bind(folder/name,old['bindings'][str(folder/name)])
    for path,h in old['bindings'].items():
        if path.endswith('.py') or path.endswith('reference_scaling.json'):bind(path,h)
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(ROOT) in str(f) and str(f).endswith('.py'):bind(f)
    for f in Path(__file__).parent.glob('*.py'):bind(f)
    for f in gate.glob('*.json'):bind(f)
    jobs=[]
    for pid,row in parents.items():
        rng=np.random.default_rng(np.random.SeedSequence([2026111307,row['train_seed']]))
        plan={str(u):[dict(seed=int(rng.integers(1,2**31-1)),times=sorted(map(int,rng.choice(np.arange(1,1001),8,replace=False)))) for _ in range(4)] for u in range(1,5)}
        # Same training seed shares full trajectory/query uniforms and feedback
        # times across O/A; different policies need not visit the same Xt.
        sec=(20*({'IMDB':480,'PubMed':650}[row['dataset']])+450)*1.25
        jobs.append(dict(id=pid+'_transport',parent=pid,dataset=row['dataset'],train_seed=row['train_seed'],architecture=row['architecture'],reward='transport',terminal_plan=plan,cached_complete=None,planning_seconds=sec,new_updates=4))
    p=dict(start_epoch=start,science_stop_epoch=start+5.5*3600,delivery_epoch=start+6*3600,
        max_GPU_hours=36.,max_GPUs=8,allowed_GPU_indices=list(range(8)),preflight_GPU_hours=0.,
        jobs=jobs,parents=parents,reused_graphs=reused,sample_seeds=old['sample_seeds'],terminal_updates=4,bindings=bindings,
        frozen=dict(lr=1e-5,clip_gradient=1.,loss_outer_scale=.01,trajectories_per_update=4,feedback_times_per_trajectory=8,EMA=.99,readout='parent EMA3000 / fine-tuned final raw',T=1000,R=1,
            feedback='LOO terminal costs, no zscore, all legal-slot log likelihood sum / (K*8), unchanged eager biased estimator',
            score='raw_joint_degree_heat_W2_squared_v1; no independent edge-count term; scales fitted only to reference',trainable='all original parameters; no new module/freeze/optimizer changes'),
        counts=dict(base_models_reused=8,base_graphs_reused=32,new_feedback_models=8,new_feedback_updates=32,new_training_chains=128,new_evaluation_chains=32,total_evaluation_graphs=64),
        scope='O / A / O+transport / A+transport, IMDB0411/0412 and PubMed0421/0422, four existing evaluation seeds',
        gate='CPU gate passed; strict asset/schema/readout binding; finite O/A temporary GPU update; full state/RNG reload; EMA restoration; reverse coverage; complete 8-job queue fits wall and GPU budget. No short-test quality hurdle.',
        primary_metrics=old['primary_metrics'],secondary_metrics=old['secondary_metrics'],
        limitations=['development seeds and graphs, not independent confirmation','warm-start full-chain fine-tuning, not from-scratch CE3000 reruns','parent EMA parameters initialize final-raw candidates; EMA also saved, not selected by outcome',
            'output package changes score and K2 to K4 versus historical typed; not a score-only causal contrast','extra feedback compute; no matched CE-continuation control',
            'fixed four feedback updates does not demonstrate convergence','CUDA re-computation bitwise equality remains uncertified','same noise uniforms/query plans do not mean same graph states',
            'score is not injective over graphs and does not guarantee exact motifs or connectivity'],
        budget=dict(historical_chain_seconds={'IMDB':480,'PubMed':650},margin=1.25,per_job_extra_seconds_before_margin=450,
            estimated_GPU_hours=sum(j['planning_seconds'] for j in jobs)/3600,predicted_eight_card_minutes=max(j['planning_seconds'] for j in jobs)/60,
            accounting='device process launch through exit incl load/save/wait; preflight included; CPU evaluation after release'),
        conclusion_rule='All metrics/graphs retained. Describe paired per-seed benefits/costs and interaction; no scalar winner or posthoc endpoint selection.')
    assert p['budget']['estimated_GPU_hours']+.3<36 and time.time()+max(j['planning_seconds'] for j in jobs)<p['science_stop_epoch']
    base.write(out/'protocol.json',p)
    with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in bindings:
            if path.endswith('.py'):archive.write(path,str(Path(path).relative_to(ROOT)))
    print(json.dumps(dict(output=str(out),counts=p['counts'],budget=p['budget'],science_stop=p['science_stop_epoch'],delivery=p['delivery_epoch']),indent=2))
if __name__=='__main__':main()
