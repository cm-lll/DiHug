"""One-shot authorized launcher activation. Writes a fresh immutable 140-minute ledger."""
import os,pathlib,json,time,hashlib,sys,datetime
ROOT=pathlib.Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
    control=pathlib.Path(sys.argv[1]).resolve()
    if (control/'ledger.json').exists() or (control/'approval.json').exists() or (control/'run').exists():raise FileExistsError('already activated: no restart or budget reset')
    template=json.loads((ROOT/'formal_preparation/authorized_execution.json').read_text())
    binding=json.loads((ROOT/'formal_preparation/final_source.json').read_text())
    for rel,digest in binding['files'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('frozen execution code differs: '+rel)
    # This entry is the formal launcher activation. All subsequent setup/checks count.
    start=time.time();tz=datetime.timezone(datetime.timedelta(hours=8))
    ledger=dict(start_epoch=start,deadline_epoch=start+8400,budget_seconds=8400,
                start_beijing=datetime.datetime.fromtimestamp(start,tz).isoformat(),
                deadline_beijing=datetime.datetime.fromtimestamp(start+8400,tz).isoformat(),
                timezone='Asia/Shanghai',scope='initialization, training, sampling, evaluation, cleanup and delivery',
                no_budget_reset=True,previous_expired_ledgers_preserved=True)
    with (control/'ledger.json').open('x') as f:json.dump(ledger,f,indent=2)
    template['ledger_sha256']=sha(control/'ledger.json')
    with (control/'approval.json').open('x') as f:json.dump(template,f,indent=2)
    (control/'launcher.pid').write_text(str(os.getpid())+'\n')
    cmd=[sys.executable,str(ROOT/'package/launch.py'),'formal','--ledger',str(control/'ledger.json'),'--approval',str(control/'approval.json'),'--protocol',str(ROOT/'protocol.json'),'--dictionary-dir',str(ROOT/'evidence'),'--cost',str(ROOT/'evidence/cost_proposal.json'),'--output',str(control/'run'),'--gpu-uuids']+template['resource_authorization']['allowed_uuids']
    (control/'command.json').write_text(json.dumps(cmd,indent=2))
    print(json.dumps(dict(activation=ledger,source_sha256=template['source_sha256'],command=cmd)),flush=True)
    os.execv(sys.executable,cmd)
if __name__=='__main__':main()
