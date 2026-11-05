"""Report-only postprocessor. CUDA disabled externally; never calls a model."""
import os,sys,time,json,traceback,csv,statistics,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.structure_output_v1.formal_runtime import OUT,BASE,CONTROL,TASKS,SAMPLES,read,save,sha,now,torch,g

def finalize():
    from experiments.structure_output_v1 import evaluate_formal,summarize_formal
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():raise RuntimeError('CPU report unexpectedly CUDA initialized')
    (OUT/'temporary').mkdir(exist_ok=True)
    evaluate_formal.main();summarize_formal.main()
    integrity=[]
    for task in TASKS:
        folder=OUT/'run'/task;row=dict(task=task)
        try:
            z=torch.load(folder/'initial.pt',map_location='cpu');old=torch.load(CONTROL/'run'/task/'initial.pt',map_location='cpu')
            extra=set()
            row['shared_initial_exact']=all(torch.equal(v,z['model'][k]) for k,v in old['model'].items())
            row['only_declared_new_parameters']=set(z['model'])-set(old['model'])==extra
            row['extra_zero']=True # no extra parameters relative to the enhanced-input control
            row['initial_RNG_exact']=g.digest(z['rng'])==g.digest(old['rng'])
            row['optimizer_fresh']=not z['optimizer']['state'];row['EMA0']=z['ema']['updates']==0
            final=torch.load(folder/'final.pt',map_location='cpu');out=torch.load(folder/'readout_ema_0.99.pt',map_location='cpu')
            row['updates']=final['updates'];row['EMA_updates']=final['ema']['updates']
            row['EMA_readout_exact']=all(torch.equal(v,out['model'][k]) for k,v in final['ema']['shadow'].items())
            logs=[json.loads(s) for s in (folder/'training.jsonl').read_text().splitlines()]
            control_logs=[json.loads(s) for s in (CONTROL/'run'/task/'training.jsonl').read_text().splitlines()]
            row['all_initial_parameters_exact']=set(old['model'])==set(z['model']) and row['shared_initial_exact']
            row['control_inputs_exact']=len(control_logs)==len(logs)==3000 and all(a['input_sha256']==b['input_sha256'] and a['t']==b['t'] for a,b in zip(logs,control_logs))
            row['paired_3000_inputs']=len(logs)==3000 and all(x['paired_input_exact'] and x['full_rng_before']==x['full_rng_after'] and x['update']==i+1 for i,x in enumerate(logs))
            row['status']='complete'
            row['all_passed']=all(row[k] for k in ('shared_initial_exact','all_initial_parameters_exact','control_inputs_exact','only_declared_new_parameters','extra_zero','initial_RNG_exact','optimizer_fresh','EMA0','EMA_readout_exact','paired_3000_inputs')) and row['updates']==row['EMA_updates']==3000
            del z,old,final,out
        except BaseException as e:row.update(status='incomplete',error=repr(e),all_passed=False)
        integrity.append(row)
    source=[dict(path=p,unchanged=sha(ROOT/p)==h) for p,h in read(OUT/'source_binding.json').items()]
    save(OUT/'final_integrity.json',dict(tasks=integrity,source=source,all_passed=all(x['all_passed'] for x in integrity) and all(x['unchanged'] for x in source)))
    metrics=read(OUT/'evaluation/per_model_four_sample_mean.json')
    primary=['main_incident4_W1','main_joint_degree_TV','triangle_composition_TV','MP2_composition_TV','MP3_composition_TV','four_cycle_composition_TV']
    raw=['edges','triangles','MP2_total','MP3_total','lcc','components','isolates','type_Chemical_low_le1_fraction','type_Chemical_max','type_Chemical_second','type_Genre_max']
    compact=[r for r in metrics if r['version']=='original' and ((r['kind']=='distance' and r['metric'] in primary) or (r['kind']=='absolute_error' and r['metric'] in ['edges','triangles','MP2_total','MP3_total']) or (r['kind']=='raw' and (r['metric'] in raw or 'Chemical' in r['metric'] and any(s in r['metric'] for s in ['max','second','top5','p90','p95']))))]
    save(OUT/'analysis/compact.json',compact)
    lines=['# 固定增强输入＋结构输出辅助：四模型匹配筛查','原图主结果；每模型四采样。INPUT为同一输入包＋CE，OUTPUT仅新增冻结结构分布辅助项。两个训练seed/数据集均为开发面板，不作显著性或稳定性证明。','|数据集|训练seed|臂|指标口径|指标|均值|采样标准差|','|---|---|---|---|---|---:|---:|']
    for r in compact:lines.append('|'+ '|'.join(str(r[k]) for k in ['dataset','train_seed','arm','kind','metric','mean','sample_std'])+'|')
    (OUT/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
    leases=[read(p) for p in (OUT/'leases').glob('*.json')];end=time.time();ledger=read(OUT/'ledger.json')
    save(OUT/'DELIVERY_RECEIPT.json',dict(at=now(),wall_minutes=(end-ledger['start_epoch'])/60,GPU_hours=sum((x.get('released_epoch',end)-x['acquired_epoch']) for x in leases)/3600,
        all_released=all('released_epoch' in x for x in leases),evaluation=read(OUT/'evaluation/receipt.json'),integrity=read(OUT/'final_integrity.json')['all_passed'],no_new_GPU_calls=True))
    save(OUT/'DONE.json',dict(at=now()))

if __name__=='__main__':
    try:
        while not ((OUT/'launcher_summary.json').exists() and ((CONTROL/'DONE.json').exists() or (CONTROL/'report_failure.json').exists())):
            if time.time()>read(OUT/'ledger.json')['deadline_epoch']-120:break
            time.sleep(10)
        finalize()
    except BaseException as e:
        save(OUT/'report_failure.json',dict(at=now(),error=repr(e),traceback=traceback.format_exc()));raise
