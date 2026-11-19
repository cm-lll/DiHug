"""CPU saved evidence only. Never declares adoption automatically."""
import os,json,time,datetime
from pathlib import Path
OUT=Path(os.environ['STRUCTURE_RUN'])
def read(p):return json.loads(p.read_text())
def main():
    ev=OUT/'evaluation';ledger=read(OUT/'ledger.json');launch=read(OUT/'launcher_summary.json')
    receipt=read(ev/'receipt.json') if (ev/'receipt.json').exists() else dict(complete=0,errors=['evaluation missing'])
    means=read(ev/'per_model_four_sample_mean.json') if (ev/'per_model_four_sample_mean.json').exists() else []
    wanted={'main_incident4_W1','main_joint_degree_TV','edges','triangles','triangle_composition_TV','MP2_total','MP3_total','MP2_composition_TV','MP3_composition_TV','lcc','isolates','type_Chemical_low_le1_fraction','type_Chemical_maximum','type_Chemical_second'}
    rows=[r for r in means if r['version']=='original' and r['metric'] in wanted]
    result=dict(finished_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),elapsed_minutes=(time.time()-ledger['start_epoch'])/60,GPU_hours=launch['GPU_hours'],new_models_expected=8,new_graphs_expected=32,reused_graphs=32,all_graphs_complete=receipt['complete']==64,worker_returncodes=launch['returncodes'],evaluation=receipt,all_owned_workers_released=True,within_delivery_deadline=time.time()<=ledger['deadline_epoch'],ordinary_recompute_tolerance_passed=False,adopted=False,followup_authorized=False)
    (OUT/'delivery.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lookup={(r['dataset'],r['train_seed'],r['arm'],r['kind'],r['metric']):r['mean'] for r in means if r['version']=='original'}
    screens=[]
    for ds,seeds in [('IMDB',[2026110411,2026110412]),('PubMed',[2026110421,2026110422])]:
        for seed in seeds:
            for arch in ['O','A']:
                evidence=[]
                for kind,metric in [('distance','main_incident4_W1'),('distance','main_joint_degree_TV'),('absolute_error','triangles'),('raw','lcc')]:
                    control=lookup.get((ds,seed,arch,kind,metric));candidate=lookup.get((ds,seed,arch+'_RC',kind,metric))
                    if control is None or candidate is None:status='missing'
                    elif metric=='lcc':status='cost' if candidate<control-.05*192 else 'within_screen'
                    else:status='cost' if candidate>1.1*control+1e-12 else 'within_screen'
                    evidence.append(dict(kind=kind,metric=metric,control=control,candidate=candidate,status=status))
                screens.append(dict(dataset=ds,train_seed=seed,architecture=arch,major_cost_screen=evidence,
                    note='predeclared development screen; passing is not adoption, statistical significance or proof of generalized stability'))
    (OUT/'quality_screen.json').write_text(json.dumps(screens,ensure_ascii=False,indent=2)+'\n')
    lines=['# 联合连接输出目标与CN四格对照：结果','',json.dumps(result,ensure_ascii=False),'',
        'O/A/O+RC/A+RC为本轮四格；O/A旧结果复用，RC同时保留同节点关系度、三角参与和不回溯关系二步连接摘要。',
        'RC是硬条件图的类型内联合分布评分；均匀1024边无放回条件差分估计，CE权重不变，辅助0.1。不是RolX，也不保证闭合伙伴身份或连通。未自动采用。下表SD为同模型四采样波动；跨训练seed另表。',
        '两训练seed属于开发面板。普通GPU重复容差未通过保留；不宣称逐位训练轨迹相同。',
        '', '|数据|seed|臂|口径|指标|均值|采样SD|','|---|---:|---|---|---|---:|---:|']
    for r in rows:lines.append('|'+ '|'.join(str(r[k]) for k in ['dataset','train_seed','arm','kind','metric','mean','sample_std'])+'|')
    (OUT/'RESULT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

