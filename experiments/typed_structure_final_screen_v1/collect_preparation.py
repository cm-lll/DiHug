"""Recompute CPU-only pairing/lineage/cost receipts from existing engineering files."""
import argparse,pathlib,json,hashlib,subprocess,math,statistics,sys
import numpy as np
import torch
from runtime_contract import ROOT,jobs,read,save,sha,source_binding
sys.path.insert(0,str(ROOT/'package'))
from prepare import load_case
from normal2 import pair_inputs
from probe import input_hash,equal

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--workspace',required=True);a=p.parse_args();out=pathlib.Path(a.results);ws=pathlib.Path(a.workspace)
    engineering=out/'engineering';proofs={};params={};compute=[]
    for dataset in ('IMDB','PubMed'):
        cells={arm:torch.load(engineering/(dataset+'_'+arm)/'initial.pt',map_location='cpu') for arm in ('A','C','D','E')}
        checks={};inp={};rng={}
        for arm,x in cells.items():
            pr=read(engineering/(dataset+'_'+arm)/'initialization.json');params[dataset+'_'+arm]=pr['construction']['parameters'];rng[arm]=pr['rng_reset']
            assert x['updates']==x['ema']['updates']==0 and not x['optimizer']['state']
            assert all(hashlib.sha256(v.numpy().tobytes()).hexdigest()==pr['construction']['tensor_sha256'][k] for k,v in x['model'].items())
            checks[arm]={}
            for name,v in x['model'].items():
                y=cells['E']['model'][name]
                if name=='joint_head.0.weight' and v.shape!=y.shape:y=y[:,:v.shape[1]]
                assert v.numpy().tobytes()==y.contiguous().numpy().tobytes(),(dataset,arm,name)
                checks[arm][name]=True
            case,cfg=load_case(dataset);cfg=dict(cfg,seed=read(ROOT/'protocol.json')['train_seeds'][dataset]);g=np.random.default_rng()
            g.bit_generator.state=x['rng']['numpy'];inp[arm]=[input_hash(pair_inputs(case,cfg,u,g)) for u in (1,2,3)]
        assert all(rng[a]==rng['A'] for a in rng)
        assert all(inp[a]==inp['A'] for a in inp)
        proofs[dataset]=dict(shared_tensor_bytes=checks,training_rng=rng,first_three_inputs=inp,status='passed')
    save(out/'pairing_audit.json',dict(status='passed',datasets=proofs,parameter_counts=params,formal_initials_created=False,engineering_initials_only=True))
    oldroots=[ws/'results/typed_structure_192_oabc_formal_v1_20261028T090624/run',ws/'results/typed_structure_192_oabc_seed2_v1_20261028T110507/run',ws/'results/typed_structure_192_de_formal_v1_20261028T140346/run']
    for task in jobs()['training']:
        d,arm=task['dataset'],task['arm'];record=read(engineering/(d+'_'+arm)/'result.json');q=read(engineering/(d+'_'+arm)/'query_cost.json')
        assert record['status']=='passed' and record['optimizer_step_calls']==2
        histories=[]
        for root in oldroots:
            for path in root.glob(d+'*'+arm):
                log=path/'training.jsonl'
                if not log.exists():continue
                vals=[json.loads(x)['normal2_seconds'] for x in log.read_text().splitlines()][5:]
                chainfiles=list(path.glob('sample_*/binding.json'));chains=[read(f)['seconds'] for f in chainfiles]
                if len(chains)!=2:continue
                histories.append(dict(path=str(path),training_log_sha256=sha(log),normal2_mean=float(np.mean(vals)),normal2_p95=float(np.percentile(vals,95)),chains_seconds=chains,chain_bindings=[dict(path=str(f),sha256=sha(f)) for f in chainfiles]))
        if not histories:raise ValueError('missing historical cost for '+d+arm)
        unit=max(record['cost']['normal2_with_ema_seconds'],max(h['normal2_p95'] for h in histories)+record['cost']['ema_overhead_seconds'])
        chain=max(max(h['chains_seconds']) for h in histories)
        validation=4*(record['validation_raw']['seconds']+record['validation_ema']['seconds'])
        saves=6*record['save_seconds'];construction=20.
        compute.append(dict(task=task,history=histories,engineering=record['cost'],query=q,normal2_planning_seconds=unit,
            training_3000_seconds=3000*unit,chain_planning_seconds=chain,two_chains_seconds=2*chain,validation_seconds=validation,save_seconds=saves,construction_and_load_allowance_seconds=construction,
            phase_train_seconds=3000*unit+validation+saves+construction,phase_sampling_seconds=2*chain+10,peak_allocated_mib=record['peak_allocated_mib'],peak_reserved_mib=record['peak_reserved_mib']))
    # Actual launcher has a barrier after ALL training; then serial chains per GPU.
    train=max(x['phase_train_seconds'] for x in compute);sample=max(x['phase_sampling_seconds'] for x in compute);body=train+sample
    concurrent_extra=.15*body;eval_seconds=300.;delivery=1200.;fault=900.;planned=body+concurrent_extra+eval_seconds+delivery+fault
    requested=math.ceil(planned/600)*600
    cost=dict(status='passed',proposal_only=True,formal_authorized=False,source_sha256=source_binding()[1],schedule=jobs(),cells=compute,
        training_phase_seconds=train,sampling_phase_seconds=sample,compute_body_seconds=body,concurrency_planning_extra_seconds=concurrent_extra,
        evaluation_LCC_seconds=eval_seconds,delivery_reserve_seconds=delivery,fault_allowance_seconds=fault,planned_seconds=planned,requested_budget_seconds=requested,
        uncertainty='15% planning margin is not measured eight-way performance. Historical full chains use old1000 raw readouts; new3000 EMA state evolution is unmeasured. One-step probes are not stable throughput.',
        stop_rule='new approved formal start+budget, science stops20minutes before deadline; no automatic resume,restart,redraw or changed scale')
    save(ROOT/'cost.json',cost);save(out/'cost.json',cost)
    # Limit seed scan to frozen candidate scientific runs, not arbitrary timestamps.
    scopes=oldroots+[ws/'results/pubmed_o_multi_ema_formal_v1_20261028T222745/run'];hits=[];checked=0
    for root in scopes:
        for file in root.rglob('binding.json'):
            checked+=1;s=file.read_text()
            for seed in (2026102911,2026102912):
                if str(seed) in s:hits.append(dict(file=str(file),seed=seed))
    save(out/'seed_audit.json',dict(scopes=list(map(str,scopes)),binding_records_checked=checked,seeds=[2026102911,2026102912],effect_observation_conflicts=hits,limitation='local sealed candidate experiment records, not an assertion about unknown remote runs'))
    if hits:raise ValueError('seed conflict')
    print(json.dumps(dict(status='passed',parameters=params,body_minutes=body/60,planned_minutes=planned/60,suggested_minutes=requested/60),indent=2))
if __name__=='__main__':main()
