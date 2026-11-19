"""Read saved evaluation only; retain complete evidence and write compact delivery."""
import os,json,time,datetime
from pathlib import Path
OUT=Path(os.environ['STRUCTURE_RUN'])
def read(p):return json.loads(p.read_text())
def main():
    ev=OUT/'evaluation';ledger=read(OUT/'ledger.json');launch=read(OUT/'launcher_summary.json')
    receipt=read(ev/'receipt.json') if (ev/'receipt.json').exists() else {'complete':0,'errors':['evaluation receipt missing']}
    means=read(ev/'per_model_four_sample_mean.json') if (ev/'per_model_four_sample_mean.json').exists() else []
    wanted={'main_incident4_W1','joint_degree_TV','edges','triangles','components','isolates','lcc_nodes'}
    rows=[r for r in means if r['version']=='original' and (r['metric'] in wanted or any(k in r['metric'].lower() for k in ['joint','triangle','mp2','mp3','chemical','lcc']))]
    status=dict(finished_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),elapsed_minutes=(time.time()-ledger['start_epoch'])/60,GPU_hours=launch['GPU_hours'],new_models=4,new_graphs_expected=16,all_graphs_complete=receipt['complete']==32,worker_returncodes=launch['returncodes'],evaluation=receipt,all_owned_workers_released=True,within_delivery_deadline=time.time()<=ledger['deadline_epoch'],ordinary_recompute_tolerance_passed=False,adopted=False,followup_authorized=False)
    (OUT/'delivery.json').write_text(json.dumps(status,indent=2,ensure_ascii=False)+'\n')
    lines=['# 闭合上下文CE加权匹配实验自动收尾','',f"原图/LCC评价完整图数：{receipt['complete']}/32（16新图＋16复用PB）。",f"累计占卡 {launch['GPU_hours']:.6f} GPU小时；当前累计墙钟 {status['elapsed_minutes']:.2f} 分钟。",'','原A＋CE＋PB与A＋上下文加权CE＋PB只改变冻结CE权重；保留全部seed和结构代价。以下为每模型四采样均值及采样标准差；不是四次独立训练。','普通GPU重复数值限制保留；不自动采纳、不追加实验。完整CSV/JSON位于evaluation。','','|数据|训练seed|臂|指标口径|指标|均值|采样SD|','|---|---:|---|---|---|---:|---:|']
    for r in rows:lines.append('|'+ '|'.join(str(r[k]) for k in ['dataset','train_seed','arm','kind','metric','mean','sample_std'])+'|')
    if receipt.get('errors'):lines+=['','失败/缺项：',json.dumps(receipt['errors'],ensure_ascii=False,indent=2)]
    (OUT/'RESULT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(status,ensure_ascii=False))
if __name__=='__main__':main()

