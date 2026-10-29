"""EMA runtime identities and bindings; deliberately no torch/GPU import."""
import hashlib,json,pathlib,os
ROOT=pathlib.Path(__file__).resolve().parent
PROTOCOL=json.loads((ROOT/'protocol.json').read_text())
TRAIN_SEEDS=PROTOCOL['train_seeds']; ARMS=tuple(PROTOCOL['arms'])
DECAYS={'ema_0.99':.99};READOUTS=('ema_0.99',);SAMPLES=tuple(PROTOCOL['sample_seeds'])
UPDATES=3000;RESERVE=1200
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(pathlib.Path(path).read_text())
def save(path,value):
    path=pathlib.Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(tmp,path)
def cell(dataset,seed,arm):
    if TRAIN_SEEDS.get(dataset)!=seed or arm not in ARMS:raise ValueError('unfrozen task')
    return f'{dataset}_s{seed}_{arm}'
def jobs():
    train=[];sample=[]
    for slot,(dataset,arm) in enumerate((d,'E') for d in ('PubMed','IMDB')):
        seed=TRAIN_SEEDS[dataset];identity=cell(dataset,seed,arm)
        common=dict(dataset=dataset,arm=arm,train_seed=seed,slot=slot)
        train.append(dict(**common,id=identity,mode='train',directory=identity,updates=UPDATES))
        for s in SAMPLES:
            sample.append(dict(**common,id=f'{identity}_ema_0.99_{s}',mode='sample',readout='ema_0.99',sample_seed=s,directory=f'{identity}/ema_0.99/sample_{s}',T=1000,depends_on=identity))
    return dict(training=train,sampling=sample,policy='two cards: slot0 PubMed E-earlyCN, slot1 IMDB E-earlyCN; then two serial final EMA chains per card')
def source_binding():
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file()
           and not any(x in p.parts for x in ('__pycache__','.pytest_cache','historical_multi_ema','evidence','tests'))
           and p.suffix in ('.py','.json','.npz') and p.name not in ('cost.json','release.json','source_manifest.json')}
    return files,hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
def dictionary_expected(dataset):
    return read(ROOT/'asset_bindings.json')['dictionaries'][dataset]
def verify_assets():
    for name,h in read(ROOT/'asset_bindings.json')['files'].items():
        if sha(ROOT/name)!=h:raise ValueError('frozen asset changed: '+name)
    bindings=read(ROOT/'baseline_bindings.json')
    for b in list(bindings['initial'].values())+[bindings['old_evaluation']]:
        if sha(b['path'])!=b['sha256']:raise ValueError('sealed external baseline changed')
def readout_path(run,job):
    if job['readout']!='ema_0.99':raise ValueError('only final EMA generates')
    return pathlib.Path(run)/cell(job['dataset'],job['train_seed'],job['arm'])/'readout_ema_0.99.pt'
def verify_readout(run,job):
    folder=pathlib.Path(run)/cell(job['dataset'],job['train_seed'],job['arm']);receipt=read(folder/'readouts.json')
    if (receipt['dataset'],receipt['arm'],receipt['train_seed'],receipt['updates'])!=(job['dataset'],job['arm'],job['train_seed'],UPDATES):raise ValueError('readout identity/count')
    p=readout_path(run,job)
    if sha(p)!=receipt['files'][job['readout']]['sha256'] or sha(folder/'final.pt')!=receipt['checkpoint_sha256']:raise ValueError('readout/checkpoint bytes differ')
    if read(folder/'summary.json')['status']!='complete':raise ValueError('training incomplete')
    return p,receipt
def approval_gate(path,cost_path,output):
    a=read(path);c=read(cost_path);verify_assets()
    if a.get('user_formal_authorized') is not True or a.get('independent_review')!='passed' or not a.get('authorization_reference'):raise PermissionError('new formal authorization and review required')
    if a['source_sha256']!=source_binding()[1] or a['protocol_sha256']!=sha(ROOT/'protocol.json'):raise ValueError('source/protocol binding')
    if a['updates']!=UPDATES or a['cost_sha256']!=sha(cost_path) or a['budget_seconds']!=c['requested_budget_seconds']:raise ValueError('budget/count/cost binding')
    if pathlib.Path(a['output']).resolve()!=pathlib.Path(output).resolve():raise ValueError('output binding')
    if c['source_sha256']!=a['source_sha256'] or c['schedule']!=jobs() or c['status']!='passed':raise ValueError('cost/source/queue mismatch')
    if c['planned_seconds']>a['budget_seconds']:raise ValueError('budget insufficient')
    if a['asset_bindings_sha256']!=sha(ROOT/'asset_bindings.json'):raise ValueError('external asset approval binding')
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
