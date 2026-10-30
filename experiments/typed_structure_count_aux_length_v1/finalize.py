"""CPU delivery from saved graphs/metrics/logs; no learned-model invocation."""
import pathlib,json,time,hashlib,csv,datetime,zipfile
from runtime_contract import read,save,sha,jobs,source_binding,ROOT
def main(run,deadline):
    run=pathlib.Path(run);records=read(run/'unified_evaluation.json') if (run/'unified_evaluation.json').exists() else {}
    receipt=read(run/'launcher_summary.json');out=run.parent;manifest=[]
    for p in sorted(run.rglob('*')):
        if not p.is_file() or 'tmp' in p.parts:continue
        if time.time()>=deadline-15:break
        if p.suffix in ('.pt','.npz','.json','.jsonl','.csv'):
            manifest.append(dict(path=str(p),relative_path=str(p.relative_to(run)),bytes=p.stat().st_size,sha256=sha(p)))
    save(out/'payload_manifest.json',manifest)
    taskrows=[]
    for j in jobs()['training']:
        folder=run/j['directory'];s=read(folder/'summary.json') if (folder/'summary.json').exists() else {'status':'not_started'}
        taskrows.append(dict(job=j,status=s,restore=read(folder/'restore_receipt.json') if (folder/'restore_receipt.json').exists() else None))
    validation=[];training=[]
    for task in taskrows:
        j=task['job'];folder=run/j['directory'];identity={k:j[k] for k in ['dataset','train_seed','arm']}
        if (folder/'validation.json').exists():
            for v in read(folder/'validation.json'):
                for kind in ['raw','ema_0.99']:
                    for row in v.get(kind,{}).get('rows',[]):
                        validation.append(dict(**identity,update=v['updates'],readout=kind,**row))
        if (folder/'training.jsonl').exists():
            for line in (folder/'training.jsonl').read_text().splitlines():
                try:
                    row=json.loads(line)
                    training.append(dict(**identity,update=row['update'],input_sha256=row['input_sha256'],t=row['t'],loss=row['loss'],ce=row['ce'],auxiliary=row['auxiliary'],weighted_auxiliary=row['weighted_auxiliary'],seconds=row['total_update_seconds']))
                except json.JSONDecodeError:continue
    save(out/'validation_panel.json',validation)
    if training:
        with (out/'training_curve.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(training[0]));w.writeheader();w.writerows(training)
    # Task metadata is nested under ``job``; status/restore are sibling fields.
    # The old flat lookup aborted delivery after successful science/evaluation.
    by={(x['job']['dataset'],x['job']['train_seed'],x['job']['arm']):x for x in taskrows}
    input_pairs=[]
    for ds,seeds in read(ROOT/'protocol.json')['train_seeds'].items():
        for seed in seeds:
            a={x['update']:x['input_sha256'] for x in training if (x['dataset'],x['train_seed'],x['arm'])==(ds,seed,'CE')}
            b={x['update']:x['input_sha256'] for x in training if (x['dataset'],x['train_seed'],x['arm'])==(ds,seed,'CE_COUNT')}
            common=a.keys()&b.keys()
            input_pairs.append(dict(dataset=ds,train_seed=seed,common_updates=len(common),all_common_match=all(a[x]==b[x] for x in common),complete_pair=a==b and len(a)==({'IMDB':2500,'PubMed':5500}[ds])))
    graphcount=sum(i.get('status')=='complete' for i in records.get('inventory',[]))
    summary=dict(status=receipt['status'],task_status=taskrows,complete_graphs=graphcount,expected_graphs=24,
        new_updates=sum(max(0,t['status'].get('completed_updates',500)-500) for t in taskrows),expected_new_updates=32000,
        input_pairing=input_pairs,science_start=read(run/'activation.json')['start_epoch'],finished_epoch=time.time(),
        source_sha256=source_binding()[1],primary_failures=receipt.get('primary_failures',[]),
        missing=records.get('missing_identities',[]),formal_extra_training_authorized=False)
    save(out/'completion.json',summary)
    lines=['# 训练长度与数量辅助配对结果','',
        f"完成新增更新{summary['new_updates']}/32000，完整链{graphcount}/24。状态：{summary['status']}。原图为主，LCC单列。",
        '以下为每个训练seed两图均值；缺失或未定义值保留null。负的误差差表示数量项误差更小；原始数量差不能直接解释为改善。',
        '','|数据集|训练seed|步数|版本|指标|CE_COUNT−CE误差差|','|---|---|---|---|---|---|']
    chosen=['edges','family1_edges','family5_edges','main_incident4_W1','main_joint_degree_TV','triangles','triangle_composition_TV','LCC','MP2_composition_TV','MP3_composition_TV']
    for x in records.get('paired_means',[]):
        if x.get('metric') not in chosen or x.get('version')!='original':continue
        if x.get('group') not in ['absolute_error','distribution_error']:continue
        lines.append('|'+ '|'.join(str(x.get(k)) for k in ['dataset','train_seed','checkpoint','version','metric','difference'])+'|')
    lines+=['','## PubMed训练长度3000→6000','|训练seed|组别|指标|6000−3000误差差|','|---|---|---|---|']
    for x in records.get('length_means',[]):
        if x.get('dataset')!='PubMed' or x.get('version')!='original' or x.get('metric') not in chosen or x.get('group') not in ['absolute_error','distribution_error']:continue
        lines.append('|'+ '|'.join(str(x.get(k)) for k in ['train_seed','arm','metric','difference'])+'|')
    lines+=['','数量更准不能代表结构全部修复；更长训练不认证收敛。两个训练seed仅支持探索判断；无本轮HGEN直接对照。完整逐图、配对与长度比较见run/derived_lcc_evaluation各CSV以及run/unified_evaluation.json。所有失败、缺图、null和LCC类型流失均保留。','',
        '本票结束后停止，不自动继续9000、补抽、更换lambda或新增实验。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # Lightweight archive excludes scientific state tensors; their full paths
    # and hashes are in payload_manifest.json. Never copy system tmp.
    if time.time()<deadline-30:
        arc=out/'review_metadata.zip'
        with zipfile.ZipFile(arc,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
            for p in sorted(run.rglob('*')):
                if time.time()>=deadline-15:break
                if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.log','.npz') and 'tmp' not in p.parts:z.write(p,'run/'+str(p.relative_to(run)))
            for name in ['completion.json','REPORT.md','payload_manifest.json','validation_panel.json','training_curve.csv']:
                p=out/name
                if p.exists():z.write(p,name)
        save(out/'archive_receipt.json',dict(path=str(arc),sha256=sha(arc),bytes=arc.stat().st_size,checkpoint_tensors_excluded=True))
    return summary
