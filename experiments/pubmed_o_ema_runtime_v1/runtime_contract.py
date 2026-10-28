"""EMA runtime identities and bindings; deliberately no torch/GPU import."""
import hashlib,json,pathlib,os
ROOT=pathlib.Path(__file__).resolve().parent
SEEDS=(2026102812,2026102833)
SAMPLES=(2026102821,2026102822)
BUDGET=7800
RESERVE=1200
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(pathlib.Path(path).read_text())
def save(path,value):
    path=pathlib.Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(tmp,path)
def cell(seed):
    if seed not in SEEDS:raise ValueError('unknown training seed')
    return 'PubMed_s%d_O'%seed
def jobs():
    train=[];sample=[]
    for slot,seed in enumerate(SEEDS):
        train.append(dict(id=cell(seed),mode='train',train_seed=seed,slot=slot,directory=cell(seed),updates=1000))
        for r in ('raw','ema'):
            for s in SAMPLES:
                sample.append(dict(id=f'{cell(seed)}_{r}_{s}',mode='sample',train_seed=seed,readout=r,sample_seed=s,
                                   slot=slot,directory=f'{cell(seed)}/{r}/sample_{s}',T=1000,depends_on=cell(seed)))
    return dict(training=train,sampling=sample,policy='two GPUs; both training jobs finish before sampling; four serial chains per GPU')
def source_binding():
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file()
           and '__pycache__' not in str(p) and '.pytest_cache' not in str(p)
           and ((p.suffix=='.py') or p==ROOT/'protocol.json' or p==ROOT/'initial_assets.json'
                or p==ROOT/'reference.json' or (ROOT/'package' in p.parents and p.suffix in ('.json','.npz')))}
    return files,hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
def asset(seed):
    return next(x for x in read(ROOT/'initial_assets.json')['records'] if x['train_seed']==seed)
def initial_path(seed,overrides=None):
    a=asset(seed);p=pathlib.Path((overrides or {}).get(str(seed),a['initial_path']))
    if sha(p)!=a['initial_sha256_from_sealed_receipt']:raise ValueError('initial checkpoint hash differs')
    return p
def readout_path(run,seed,kind):
    if kind not in ('raw','ema'):raise ValueError('readout')
    return pathlib.Path(run)/cell(seed)/('readout_'+kind+'.pt')
def verify_readout(run,job):
    folder=pathlib.Path(run)/cell(job['train_seed']);receipt=read(folder/'readouts.json')
    if receipt['train_seed']!=job['train_seed'] or receipt['updates']!=1000:raise ValueError('readout identity/count')
    p=readout_path(run,job['train_seed'],job['readout'])
    if sha(p)!=receipt['files'][job['readout']]['sha256']:raise ValueError('readout bytes differ')
    if sha(folder/'final.pt')!=receipt['checkpoint_sha256']:raise ValueError('readout final checkpoint differs')
    if read(folder/'summary.json')['status']!='complete':raise ValueError('training incomplete')
    return p,receipt
def approval_gate(path,cost_path,output):
    a=read(path);c=read(cost_path)
    if a.get('user_formal_authorized') is not True or a.get('independent_review')!='passed' or not a.get('authorization_reference'):raise PermissionError('explicit formal approval missing')
    if a['source_sha256']!=source_binding()[1] or a['protocol_sha256']!=sha(ROOT/'protocol.json'):raise ValueError('source/protocol binding')
    if a['budget_seconds']!=BUDGET or a['updates']!=1000 or a['cost_sha256']!=sha(cost_path):raise ValueError('budget/count/cost binding')
    if pathlib.Path(a['output']).resolve()!=pathlib.Path(output).resolve():raise ValueError('output binding')
    if c['source_sha256']!=a['source_sha256'] or c['status']!='passed' or c['schedule']!=jobs():raise ValueError('cost/source/queue mismatch')
    if c['planned_seconds']>BUDGET:raise ValueError('budget insufficient')
    evidence=read(c['preflight_path'])
    if sha(c['preflight_path'])!=c['preflight_sha256'] or evidence['status']!='passed' or evidence['source_sha256']!=a['source_sha256']:raise ValueError('preflight evidence binding')
    for seed in SEEDS:initial_path(seed,a.get('initial_paths'))
    return a,c
def verify_ticket(ticket,job,approval,ledger,device,parent):
    if ticket.get('parent_pid')!=parent or ticket.get('job')!=job or ticket.get('device')!=device:raise PermissionError('launcher ticket mismatch')
    if ticket['approval_sha256']!=sha(approval) or ticket['ledger_sha256']!=sha(ledger):raise ValueError('launch binding differs')
    l=read(ledger);a=read(approval)
    if l['approval_sha256']!=sha(approval) or l['source_sha256']!=source_binding()[1] or a['source_sha256']!=l['source_sha256']:raise ValueError('activated source/approval mismatch')
    if l['deadline_epoch']!=l['start_epoch']+BUDGET:raise ValueError('deadline changed')
def output_check(path,protected):
    p=pathlib.Path(path).resolve()
    if p.exists():raise FileExistsError('new output only, no automatic restart')
    if any(p==q or q in p.parents or p in q.parents for q in [pathlib.Path(x).resolve() for x in protected]):raise ValueError('output overlaps source/input')
    return p
