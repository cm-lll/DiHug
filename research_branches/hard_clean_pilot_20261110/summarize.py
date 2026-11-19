"""All-task CPU summary; missing/failed conditions stay visible."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,csv
from pathlib import Path
import numpy as np

def main():
    out=Path(sys.argv[1]);p=json.loads((out/'protocol.json').read_text());tasks=[];flat=[];paired=[]
    for row in p['tasks']:
        folder=out/row['task'];receipt=folder/'receipt.json'
        r=json.loads(receipt.read_text()) if receipt.exists() else dict(status='not_started',updates=0)
        tasks.append(dict(task=row['task'],dataset=row['dataset'],seed=row['seed'],arm=row['arm'],**r))
        ev=folder/'evaluation.json'
        if not ev.exists():continue
        for v in json.loads(ev.read_text()):
            base={k:v[k] for k in ('offset','updates','readout','t','input_sha256','CE','positive_CE','negative_CE')}
            base.update(task=row['task'],dataset=row['dataset'],seed=row['seed'],arm=row['arm'])
            base.update(dict(zip(p['objective_groups'],v['hard_cost_mean'])))
            for key in ('edges','triangles','components','LCC','isolated','hub'):
                arr=[s[key] for s in v['samples']];base[key]=float(np.mean(arr));base[key+'_sample_sd']=float(np.std(arr,ddof=1))
            flat.append(base)
    for v in flat:
        if v['arm']!='HARD_MGDA':continue
        match=[w for w in flat if w['arm']=='CE' and all(w[k]==v[k] for k in ('dataset','seed','offset','readout','t','input_sha256'))]
        if len(match)!=1:continue
        w=match[0];r={k:v[k] for k in ('dataset','seed','offset','readout','t','input_sha256')}
        for k in ['CE','positive_CE','negative_CE']+p['objective_groups']+['edges','triangles','components','LCC','isolated','hub']:
            r[k+'_CE']=w[k];r[k+'_HARD']=v[k];r[k+'_difference']=v[k]-w[k]
        paired.append(r)
    for name,rows in [('tasks',tasks),('conditional_results',flat),('paired',paired)]:
        (out/(name+'.json')).write_text(json.dumps(rows,indent=2,ensure_ascii=False))
        if rows:
            keys=list(dict.fromkeys(k for r in rows for k in r))
            with (out/(name+'.csv')).open('w') as f:
                w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
    audit=[]
    for ds,seeds in [('IMDB',[2026110411,2026110412]),('PubMed',[2026110421,2026110422])]:
        for seed in seeds:
            histories=[]
            for arm in ['CE','HARD_MGDA']:
                f=out/f'{ds}_{seed}_{arm}'/'updates.jsonl'
                histories.append([json.loads(x) for x in f.read_text().splitlines()] if f.exists() else [])
            n=min(map(len,histories));equal=all(histories[0][i]['input_sha256']==histories[1][i]['input_sha256'] and histories[0][i]['main_rng_sha256']==histories[1][i]['main_rng_sha256'] for i in range(n))
            audit.append(dict(dataset=ds,seed=seed,common_updates=n,main_inputs_and_rng_equal=equal,full_pair=n==p['updates_each']))
    (out/'pairing_audit.json').write_text(json.dumps(audit,indent=2))
    done=sum(x['status']=='complete' for x in tasks)
    report=['# 真实小图一步硬结构反馈试点',f'完成任务 {done}/8。所有未启动、停止及错误见 tasks.json。',
            '这是2000步父状态的128次局部续训；原始加噪和反向核不变，没有完整扩散生成图。',
            '候选没有CE项：16个独立硬图的LOO反馈→五组参数梯度MGDA→继承AdamW。估计梯度无偏不代表MGDA方向无偏；AdamW也不保持纯梯度的共同下降保证。',
            '独立条件预测评价是同参考图新噪声，不是跨图泛化；32张条件硬图不是32次独立训练。',
            '对照为原CE同父状态、同输入流的短续训。原始检查点未覆盖。',
            '此票不自动启动长训练或完整生成。']
    for x in tasks:report.append(f"- {x['task']}: {x['status']}, 完成更新 {x.get('updates',0)}, {x.get('error','')}")
    (out/'REPORT_ZH.md').write_text('\n\n'.join(report)+'\n')
    print(json.dumps(dict(completed=done,rows=len(flat),paired=len(paired),audit=audit)))
if __name__=='__main__':main()
