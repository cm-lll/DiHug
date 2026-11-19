import datetime,json,hashlib,shutil,math
from pathlib import Path
R=Path(__file__).resolve().parents[2];H=Path(__file__).parent;O=R/'results/direct_structure_response_20261108T205816';OLD=R/'results/hard_structure_response_20261108T182154'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
O.mkdir(exist_ok=False);start=datetime.datetime.fromisoformat('2026-11-08T20:58:16+08:00').timestamp()
calibration=R/'results/direct_structure_gradient_gpu_20261108T205133/results.json';cal=read(calibration);maxratio=max(x['geometry']['all']['ratio'] for x in cal);coefficient=10**math.floor(math.log10(.25/maxratio));assert coefficient==.001
oldp=read(OLD/'protocol.json')
p=dict(start_epoch=start,science_stop_epoch=start+1020,delivery_stop_epoch=start+1200,gpu_seconds_limit=720,steps=8,coefficient=coefficient,auxiliary_draws=1,evaluation_draws=32,times=[1,500,1000],max_updates=32,max_forwards=44,max_backwards=32,aux_seeds=oldp['aux_seeds'],evaluation_seeds=oldp['evaluation_seeds'],evaluation_mc_seeds=oldp['evaluation_mc_seeds'],purpose='matched temporary parameter response only; historical CE eight-step controls reused, no formal training or T1000',coefficient_rationale=dict(rule='10**floor(log10(0.25/max four native parameter gradient ratios))',calibration=str(calibration),sha256=sha(calibration),max_ratio=maxratio,calibration_weighted_ratios=[coefficient*x['geometry']['all']['ratio'] for x in cal],note='Engineering calibration only. Fixed across datasets/steps; not a cap or clipping rule on later inputs. No outcome-based search.'),decision='Report all three time points and each parent; no adoption from eight updates. Source/state/input/nonfinite/budget faults stop. No quality gate or lambda search.',evaluation_scope='raw conditional p0 probabilities and 32 paired Bernoulli auxiliary graphs; not full reverse-chain outputs; previously viewed development inputs independent of update stream')
(O/'protocol.json').write_text(json.dumps(p,indent=2))
audit=read(OLD/'audit.json');assert audit['all_bindings_unchanged']
for path,h in read(OLD/'source_binding.json').items():assert sha(Path(path))==h
assets={a['condition']:a for a in read(OLD/'assets.json')};bindings={}
for c,a in assets.items():
    parent=Path(a['parent']);assert sha(parent)==a['sha256'];cp=OLD/f'{c}_CE_temporary.pt';assert sha(cp)==audit['local_checkpoint_sha256'][str(cp)]
    bindings[c]=dict(checkpoint=str(cp),checkpoint_sha256=sha(cp),parent_sha256=a['sha256'],parent=str(parent),fixed_inputs=a['fixed_inputs'],original_audit=str(OLD/'audit.json'))
(O/'control_binding.json').write_text(json.dumps(bindings,indent=2))
probes=[x for x in read(OLD/'probes.json') if x['arm'] in ['start','CE']];assert len(probes)==24
(O/'probes.json').write_text(json.dumps(probes,indent=2));updates=[x for x in read(OLD/'updates.json') if x['arm']=='CE'];assert len(updates)==32
(O/'control_updates.json').write_text(json.dumps(updates,indent=2))
reuse={str(OLD/'probes.json'):sha(OLD/'probes.json'),str(OLD/'updates.json'):sha(OLD/'updates.json'),str(OLD/'protocol.json'):sha(OLD/'protocol.json')}
for x in probes:
    fn=f"{x['condition']}_{x['arm']}_t{x['t']}.npz";shutil.copy2(OLD/fn,O/fn);reuse[str(OLD/fn)]=sha(OLD/fn)
(O/'reused_assets.json').write_text(json.dumps(reuse,indent=2))
sources=list(H.glob('*.py'))+[R/'research_branches/direct_structure_cpu_20261108/score.py',R/'research_branches/hard_structure_edge_direction_cpu_20261108/metrics.py',R/'research_branches/hard_structure_rebar_cpu_20261108/objective.py']
(O/'source_binding.json').write_text(json.dumps({str(x):sha(x) for x in sources},indent=2))
for f in H.glob('*.py'):compile(f.read_text(),str(f),'exec')
print(json.dumps(dict(output=str(O),coefficient=coefficient,calibration_ratios=p['coefficient_rationale']['calibration_weighted_ratios'],controls=len(bindings))))
