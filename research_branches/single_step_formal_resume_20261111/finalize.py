"""Missing-aware paired statistics, no candidate selection or automatic retry."""
import json,csv,time,statistics,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);OUT=parser.parse_args().out
p=json.loads((OUT/'protocol.json').read_text());rows=[];status=[];pairing=[]
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def csvsave(n,x):
    if not x:return
    keys=list(dict.fromkeys(k for r in x for k in r))
    with (OUT/n).open('w') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(x)
for task in p['tasks']:
    f=OUT/task['id'];rec=json.loads((f/'receipt.json').read_text()) if (f/'receipt.json').exists() else {'status':'not_started'};assert rec.get('task',task['id'])==task['id'],'task identity mismatch'
    status.append({**rec,'task':task['id']})
    if (f/'per_graph.json').exists():rows+=json.loads((f/'per_graph.json').read_text())
for i in (0,2,4,6):
    pair=[OUT/p['tasks'][j]['id']/'training.jsonl' for j in (i,i+1)];a,b=[[json.loads(x) for x in f.read_text().splitlines()] if f.exists() else [] for f in pair];checks=[]
    for x,y in zip(a,b):checks.append(x['update']==y['update'] and x['main_rng_hash']==y['main_rng_hash'] and x['records'][0]['input_hash']==y['records'][0]['input_hash'] and x['records'][0]['hard_prefix16_hash']==y['records'][0]['hard_prefix16_hash'])
    pairing.append(dict(seed=p['tasks'][i]['parent']['seed'],completed_left=len(a),completed_right=len(b),paired_prefix_all_pass=all(checks),checked=len(checks),full_pair=len(a)==len(b)==1000))
identity={'dataset','train_seed','arm','B','K','sample_seed','view'};metrics=[k for k in rows[0] if k not in identity] if rows else [];means=[];across=[];pairs=[]
for ds in ['IMDB','PubMed']:
    for view in ['original','LCC']:
        for seed in sorted({t['parent']['seed'] for t in p['tasks'] if t['parent']['dataset']==ds}):
            for arm in ['B1K64','B4K16','ordinary_CE_background']:
                x=[r for r in rows if r['dataset']==ds and r['view']==view and r['train_seed']==seed and (r['arm'].endswith(arm) if arm!='ordinary_CE_background' else r['arm']==arm)]
                rr=dict(dataset=ds,view=view,seed=seed,arm=arm,samples=len(x))
                for metric in metrics:
                    v=[r[metric] for r in x if isinstance(r[metric],(int,float))];rr[metric]=statistics.mean(v) if len(v)==4 else None;rr[metric+'_sample_SD']=statistics.stdev(v) if len(v)==4 else None
                means.append(rr)
            a,b=means[-3],means[-2]
            pairs.append(dict(dataset=ds,view=view,seed=seed,**{m:(b[m]-a[m] if b[m] is not None and a[m] is not None else None) for m in metrics}))
        for arm in ['B1K64','B4K16','ordinary_CE_background']:
            x=[r for r in means if r['dataset']==ds and r['view']==view and r['arm']==arm];rr=dict(dataset=ds,view=view,arm=arm,training_seeds=len(x))
            for metric in metrics:
                v=[r[metric] for r in x if r[metric] is not None];rr[metric+'_mean']=statistics.mean(v) if len(v)==2 else None;rr[metric+'_training_SD']=statistics.stdev(v) if len(v)==2 else None
            across.append(rr)
for n,x in [('per_graph',rows),('per_model',means),('across_training_seeds',across),('paired_B4_minus_B1',pairs)]:save(n+'.json',x);csvsave(n+'.csv',x)
save('task_status.json',status);save('training_pairing.json',pairing)
lines=['# 单步硬结构训练：输入覆盖匹配对照','',f'正式任务完成 {sum(r["status"]=="completed" for r in status)}/8；新增完整图 {sum(r["view"]=="original" and r["arm"]!="ordinary_CE_background" for r in rows)}/32。','两臂均从CE训练2000步完整状态续训1000步，只用原五硬结构风险；不是从零纯结构训练。B4每步4输入比B1多4倍曝光及前向，均64硬样本；算术平均目标及原AdamW/EMA不改。旧普通CE仅背景。','原图为主、LCC另列；两seed为筛查，不作显著性声明，不把四张采样图当独立训练。','', '|数据|seed|方案|边数误差|度W1|联合度TV|三角误差|MP3误差|LCC|','|---|---|---|---|---|---|---|---|---|']
for r in means:
    if r['view']=='original':lines.append('|'+ '|'.join(str(r.get(k)) for k in ['dataset','seed','arm','edge_error','degree_W1','joint_TV','triangle_error','MP3_error','lcc'])+'|')
lines+=['','所有均值对应完整四图；缺图不报完整均值。逐图、采样SD、跨训练seed样本SD及全部结构代价见CSV。训练输入/硬随机前缀配对检查见training_pairing.json；CUDA重复更新逐位保证仍未认证。','不同指标的方向必须分别解释，不以均值更小或图更相似宣布结构稳定性解决。没有自动追加训练、换seed或择优检查点。']
(OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
save('completion.json',dict(at_epoch=time.time(),complete_tasks=sum(r['status']=='completed' for r in status),expected_tasks=8,graphs=sum(r['view']=='original' and r['arm']!='ordinary_CE_background' for r in rows),expected_graphs=32,all_pairing_checks_pass=all(r['paired_prefix_all_pass'] for r in pairing),formal_continuation_not_from_scratch=True,automatic_followup=False))
save('result_manifest.json',{str(f.relative_to(OUT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.glob('*') if f.is_file() and f.name!='result_manifest.json'})
