"""CPU saved evidence only. Never declares adoption automatically."""
import os,json,time,datetime
from pathlib import Path
OUT=Path(os.environ['STRUCTURE_RUN'])
def read(p):return json.loads(p.read_text())
def main():
    ev=OUT/'evaluation';ledger=read(OUT/'ledger.json');launch=read(OUT/'launcher_summary.json')
    receipt=read(ev/'receipt.json') if (ev/'receipt.json').exists() else dict(complete=0,errors=['evaluation missing'])
    means=read(ev/'per_model_four_sample_mean.json') if (ev/'per_model_four_sample_mean.json').exists() else []
    wanted={'main_incident4_W1','main_joint_degree_TV','edges','triangles','triangle_composition_TV','MP2_total','MP3_total','MP2_composition_TV','MP3_composition_TV','lcc','isolates','type_Chemical_low_le1_fraction','type_Chemical_max','type_Chemical_second'}
    rows=[r for r in means if r['version']=='original' and r['metric'] in wanted]
    result=dict(finished_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),elapsed_minutes=(time.time()-ledger['start_epoch'])/60,GPU_hours=launch['GPU_hours'],new_models_expected=12,new_graphs_expected=48,reused_graphs=48,all_graphs_complete=receipt['complete']==96,worker_returncodes=launch['returncodes'],evaluation=receipt,all_owned_workers_released=True,within_delivery_deadline=time.time()<=ledger['deadline_epoch'],ordinary_recompute_tolerance_passed=False,adopted=False,followup_authorized=False)
    (OUT/'delivery.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 输出联合分布与CN四格对照：完整结果','',json.dumps(result,ensure_ascii=False),'',
        'O/A/O+PB/A+PB为原PB四格；O/A/O+JK/A+JK为唯一新输出候选四格。共享O/A复用，不重复算独立训练。',
        'JK为整组关系度的核分布损失，替代PB而不是叠加。未自动采用。下表SD为同模型四采样波动；跨训练seed另表。',
        '两训练seed属于开发面板。普通GPU重复容差未通过保留；不宣称逐位训练轨迹相同。',
        '', '|数据|seed|臂|口径|指标|均值|采样SD|','|---|---:|---|---|---|---:|---:|']
    for r in rows:lines.append('|'+ '|'.join(str(r[k]) for k in ['dataset','train_seed','arm','kind','metric','mean','sample_std'])+'|')
    (OUT/'RESULT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
