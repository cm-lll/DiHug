"""EMA runtime identities and bindings; deliberately no torch/GPU import."""
import hashlib,json,pathlib,os
ROOT=pathlib.Path(__file__).resolve().parent
PROTOCOL=json.loads((ROOT/'protocol.json').read_text())
TRAIN_SEEDS=PROTOCOL['train_seeds']; ARMS=tuple(PROTOCOL['arms'])
DECAYS={'ema_0.99':.99};READOUTS=('ema_0.99',);SAMPLES=tuple(PROTOCOL['sample_seeds'])
UPDATES=PROTOCOL['updates'];RESERVE=1800
def final_updates(dataset):return PROTOCOL['final_updates'][dataset]
def generation_checkpoints(dataset):return [u for u in PROTOCOL['generation_checkpoints'] if u<=final_updates(dataset)]
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(pathlib.Path(path).read_text())
def save(path,value):
    path=pathlib.Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(tmp,path)
def cell(dataset,seed,arm):
    if (dataset,seed,arm) not in [(x['dataset'],x['train_seed'],x['arm']) for x in PROTOCOL['tasks']]:raise ValueError('unfrozen task')
    return f'{dataset}_s{seed}_{arm}'
def jobs():
    train=[];sample=[]
    for task in PROTOCOL['tasks']:
        dataset,arm,seed,slot=(task[k] for k in ('dataset','arm','train_seed','slot'))
        identity=cell(dataset,seed,arm)
        common=dict(dataset=dataset,arm=arm,train_seed=seed,slot=slot,model='E',fusion='early')
        train.append(dict(**common,id=identity,mode='train',directory=identity,updates=final_updates(dataset),resume_updates=500))
        for u in generation_checkpoints(dataset):
            for seed_sample in SAMPLES:
                sample.append(dict(**common,id=f'{identity}_u{u}_ema_0.99_{seed_sample}',mode='sample',readout='ema_0.99',sample_seed=seed_sample,checkpoint=u,directory=f'{identity}/u{u}/ema_0.99/sample_{seed_sample}',T=1000,depends_on=identity))
    return dict(training=train,sampling=sample,policy='8 cards; resume500;IMDB3000;PubMed6000;each card generates two chains at3000,PubMed two more at6000;no global barrier')
def source_binding():
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file()
           and not any(x in p.parts for x in ('__pycache__','.pytest_cache','historical_multi_ema','historical_metadata','evidence','tests'))
           and p.suffix in ('.py','.json','.npz') and p.name not in ('cost.json','release.json','source_manifest.json','PAYLOAD_SHA256.json')}
    return files,hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
def dictionary_expected(dataset):
    return read(ROOT/'asset_bindings.json')['dictionaries'][dataset]
def verify_assets():
    for name,h in read(ROOT/'asset_bindings.json')['files'].items():
        if sha(ROOT/name)!=h:raise ValueError('frozen asset changed: '+name)
    for b in read(ROOT/'initial_bindings.json').values():
        if sha(b['path'])!=b['sha256']:raise ValueError('sealed theta0 changed')
def readout_path(run,job):
    if job['readout']!='ema_0.99' or job['checkpoint'] not in PROTOCOL['generation_checkpoints']:raise ValueError('unapproved readout')
    return pathlib.Path(run)/cell(job['dataset'],job['train_seed'],job['arm'])/f"readout_u{job['checkpoint']}_ema_0.99.pt"
def verify_readout(run,job):
    folder=pathlib.Path(run)/cell(job['dataset'],job['train_seed'],job['arm'])
    receipt=read(folder/f"readouts_u{job['checkpoint']}.json")
    if (receipt['dataset'],receipt['arm'],receipt['train_seed'],receipt['updates'])!=(job['dataset'],job['arm'],job['train_seed'],job['checkpoint']):raise ValueError('readout identity/count')
    path=readout_path(run,job)
    if sha(path)!=receipt['files'][job['readout']]['sha256'] or sha(folder/f"checkpoint_u{job['checkpoint']}.pt")!=receipt['checkpoint_sha256']:raise ValueError('readout/checkpoint bytes differ')
    return path,receipt
def approval_gate(path,cost_path,output):
    a=read(path);c=read(cost_path);verify_assets()
    if a.get('user_formal_authorized') is not True or a.get('independent_review')!='passed' or not a.get('authorization_reference'):raise PermissionError('new formal authorization and review required')
    if a['source_sha256']!=source_binding()[1] or a['protocol_sha256']!=sha(ROOT/'protocol.json'):raise ValueError('source/protocol binding')
    if a['updates']!=PROTOCOL['final_updates'] or a['cost_sha256']!=sha(cost_path) or a['budget_seconds']!=c['requested_budget_seconds']:raise ValueError('budget/count/cost binding')
    if pathlib.Path(a['output']).resolve()!=pathlib.Path(output).resolve():raise ValueError('output binding')
    if c['source_sha256']!=a['source_sha256'] or c['schedule']!=jobs() or c['status']!='passed':raise ValueError('cost/source/queue mismatch')
    if a['gpu_uuids']!=c['proposed_gpu_uuids']:raise ValueError('cost and actual physical queue differ')
    if c['planned_seconds']>a['budget_seconds']:raise ValueError('budget insufficient')
    if a['initial_bindings_sha256']!=sha(ROOT/'initial_bindings.json'):raise ValueError('initial bindings')
    if not isinstance(a.get('environment'),dict) or not all(k in a['environment'] for k in ['python_version','torch_runtime_version','packages','device_name']):raise ValueError('complete environment required')
    if a['asset_bindings_sha256']!=sha(ROOT/'asset_bindings.json'):raise ValueError('external asset approval binding')
    if a['budget_seconds']!=32400 or a['ticket_start_epoch']!=1793286828:raise ValueError('ticket clock changed')
    if a['formula_sha256']!=sha(ROOT/'formula.json') or a['lambda_count']!=.001:raise ValueError('formula approval mismatch')
    if a['parent_checkpoints_sha256']!=sha(ROOT/'parent_checkpoints.json') or a['parent_contract_sha256']!=sha(ROOT/'parent_contract.json'):raise ValueError('parent compatibility binding')
    if sha(a['wiring_receipt']['path'])!=a['wiring_receipt']['sha256'] or read(a['wiring_receipt']['path'])['status']!='passed':raise ValueError('wiring checks incomplete')
    return a,c
def verify_ticket(ticket,job,approval,ledger,device,parent):
    if ticket.get('parent_pid')!=parent or ticket.get('job')!=job or ticket.get('device')!=device:raise PermissionError('launcher ticket mismatch')
    if ticket['approval_sha256']!=sha(approval) or ticket['ledger_sha256']!=sha(ledger):raise ValueError('launch binding differs')
    l=read(ledger);a=read(approval)
    if l['approval_sha256']!=sha(approval) or l['source_sha256']!=source_binding()[1] or a['source_sha256']!=l['source_sha256']:raise ValueError('activated source/approval mismatch')
    if l['deadline_epoch']!=l['start_epoch']+a['budget_seconds']:raise ValueError('deadline changed')
def output_check(path,protected):
    p=pathlib.Path(path).resolve()
    if p.exists():raise FileExistsError('new output only, no automatic restart')
    if any(p==q or q in p.parents or p in q.parents for q in [pathlib.Path(x).resolve() for x in protected]):raise ValueError('output overlaps source/input')
    return p
