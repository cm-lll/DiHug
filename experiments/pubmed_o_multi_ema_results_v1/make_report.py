"""Export all frozen results; no model calls or choice of decay/checkpoint."""
import argparse,csv,datetime,json,pathlib,statistics,sys,time
import numpy as np

def read(p):return json.loads(pathlib.Path(p).read_text())
def table(header,rows):
    return '\n'.join(['| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |']+['| '+' | '.join(str(v) for v in r)+' |' for r in rows])
def dump_csv(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def f(v):return 'null' if v is None else f'{v:.4f}' if isinstance(v,float) else str(v)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    run=pathlib.Path(args.run);src=pathlib.Path(args.source);out=pathlib.Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir();sys.path.insert(0,str(src));import runtime_contract as c
    a=read(run/'activation.json');x=read(run/'unified_evaluation.json');end=read(run/'launcher_summary.json');assessment=x['assessment']
    assert x['status']==end['status']=='complete'
    def check():
        if time.time()>=a['deadline_epoch']-60:raise TimeoutError('original final deadline')
    graph_rows=[];long_rows=[];cell_rows=[];pair_rows=[];group_rows=[];validation=[];families=[];cost=[];train=[]
    for cell in x['cells'].values():
        for g in cell['graphs']:
            check();m=g['metrics'];row=dict(train_seed=g['train_seed'],group='old_two' if g['train_seed'] in c.OLD_SEEDS else 'new_six',readout=g['readout'],sample_seed=g['sample_seed'],status=g['status'],edges=m['edges'],signed_edge_bias=m['edges']-926,edge_absolute_error=abs(m['edges']-926),W1_incident4=m['W1_definitions']['incident4'],W1_all10=m['W1_definitions']['all10_compat'],joint_degree_TV=m['primary']['type_weighted_joint_degree_TV'],triangles=m['triangles'],triangle_absolute_error=abs(m['triangles']-m['reference_metrics']['triangles']),triangle_composition_TV=m['primary']['typed_triangle_composition_TV'],LCC=m['lcc'],isolates=m['isolates'],maximum_degree=m['maximum_degree'],components=len(m['component_sizes']),average_clustering=m['average_clustering'])
            for j in range(3):row['MP'+str(j+1)+'_total']=m['mp'][j]['total'];row['MP'+str(j+1)+'_composition_TV']=m['comparison']['mp_frequency_TV'][j]
            for j,n in enumerate(m['family_edges']):row['family_'+str(j)+'_edges']=n
            graph_rows.append(row)
            for group,values in g['values'].items():
                if group=='undefined':continue
                for name,value in values.items():long_rows.append(dict(train_seed=g['train_seed'],readout=g['readout'],sample_seed=g['sample_seed'],metric_group=group,metric=name,value=value,reason=g['values']['undefined'].get(group,{}).get(name)))
    for seed in c.SEEDS:
        for kind in c.READOUTS:
            rows=[r for r in graph_rows if r['train_seed']==seed and r['readout']==kind]
            row=dict(train_seed=seed,group='old_two' if seed in c.OLD_SEEDS else 'new_six',readout=kind)
            for k in rows[0]:
                if k not in ['train_seed','readout','sample_seed','group','status']:row[k]=sum(r[k] for r in rows)/len(rows) if all(r[k] is not None for r in rows) else None
            cell_rows.append(row)
        folder=run/c.cell(seed);v=read(folder/'validation.json');receipt=read(folder/'summary.json');rr=[json.loads(l) for l in (folder/'training.jsonl').read_text().splitlines()]
        for r in rr:train.append(dict(train_seed=seed,updates=r['update'],loss=r['loss'],gradient_norm=r['gradient_norm'],normal2_seconds=r['normal2_seconds'],total_update_seconds=r['total_update_seconds'],input_sha256=r['input_sha256'],t0=r['t'][0],t1=r['t'][1]))
        for record in v:
            for kind in c.READOUTS:
                for r in record[kind]['rows']:
                    validation.append(dict(train_seed=seed,updates=record['updates'],readout=kind,t=r['t'],input_sha256=r['input_sha256'],CE_sum_over_S=r['ce_sum_over_S'],p0_probability_sum=sum(z['expected_edges'] for z in r['family']),clean_edges=926))
                    for z in r['family']:families.append(dict(train_seed=seed,updates=record['updates'],readout=kind,t=r['t'],input_sha256=r['input_sha256'],**z))
        for kind in c.READOUTS:
            for sample in c.SAMPLES:
                s=read(folder/kind/('sample_'+str(sample))/'summary.json');cost.append(dict(train_seed=seed,readout=kind,sample_seed=sample,chain_seconds=s['finished_epoch']-s['started_epoch'],peak_allocated_mib=s['peak_allocated_mib'],peak_reserved_mib=s['peak_reserved_mib'],peak_rss_mib=s['peak_rss_mib']))
        cost.append(dict(train_seed=seed,readout='training',sample_seed=None,worker_seconds=receipt['finished_epoch']-receipt['started_epoch'],normal2_seconds=sum(r['normal2_seconds'] for r in rr),update_including_ema_seconds=sum(r['total_update_seconds'] for r in rr),four_readout_validation_seconds=sum(r[k]['seconds'] for r in v for k in c.READOUTS),peak_allocated_mib=receipt['peak_allocated_mib'],peak_reserved_mib=receipt['peak_reserved_mib'],peak_rss_mib=receipt['peak_rss_mib']))
    for kind,d in assessment['decays'].items():
        for p in d['per_seed']:
            rr={r['readout']:r for r in cell_rows if r['train_seed']==p['train_seed']}
            row=dict(train_seed=p['train_seed'],group=p['group'],readout=kind,primary=kind=='ema_0.99',raw_edge_absolute_error=p['raw']['edge_absolute_error'],ema_edge_absolute_error=p['ema']['edge_absolute_error'],edge_error_difference=p['difference'],**p['warnings'])
            for k in rr[kind]:
                if k not in ['train_seed','readout','group']:row[k+'_difference']=None if rr[kind][k] is None or rr['raw'][k] is None else rr[kind][k]-rr['raw'][k]
            pair_rows.append(row)
        for label,g in d['groups'].items():
            seeds=c.SEEDS if label=='all_eight' else c.OLD_SEEDS if label=='old_two' else c.NEW_SEEDS
            cells=[z for z in cell_rows if z['train_seed'] in seeds]
            row=dict(readout=kind,primary=kind=='ema_0.99',group=label,**{k:v for k,v in g.items() if k!='signed_bias'})
            for k in ['raw',kind]:
                rows=[z for z in cells if z['readout']==k]
                for metric in ['edge_absolute_error','signed_edge_bias','W1_incident4','joint_degree_TV','triangles','triangle_absolute_error','triangle_composition_TV','LCC','isolates','maximum_degree','MP2_composition_TV','MP3_composition_TV']:
                    row[k+'_'+metric]=sum(z[metric] for z in rows)/len(rows)
            group_rows.append(row)
    for name,rows in [('all_graph_metrics',graph_rows),('all_metric_values',long_rows),('cell_means',cell_rows),('paired_metrics',pair_rows),('group_summary',group_rows),('training_curves',train),('validation',validation),('validation_by_family',families),('costs',cost)]:dump_csv(out/(name+'.csv'),rows)
    lines=['# PubMed O多EMA：八训练seed完整结果','',f"正式起点：{datetime.datetime.fromtimestamp(a['start_epoch']).astimezone().isoformat()}。统一评价完成：{datetime.datetime.fromtimestamp(end['finished_epoch']).astimezone().isoformat()}，耗时{end['elapsed_seconds']/60:.2f}分钟。",'',
           '8条训练轨迹各1000次normal2更新，32份最终读出，64条全新T1000链，全部完成。没有失败补抽、重启、换seed、追加训练或图修复。原两个开发seed与新增六seed分开展示。主比较固定EMA0.99，另两尺度全部附报。','',
           '## 主要指标：每图相对926的绝对边数误差，先在训练seed内平均两图','',table(['训练seed','raw','EMA0.9','EMA0.97','EMA0.99'],[[seed]+[f(next(z for z in cell_rows if z['train_seed']==seed and z['readout']==k)['edge_absolute_error']) for k in c.READOUTS] for seed in c.SEEDS]),'',
           '负的EMA−raw表示误差下降；零为持平，正为误差增加。不能用平均有符号数量偏差抵消过量与不足。','',
           table(['读出','seed组','raw绝对误差','EMA绝对误差','平均配对差','中位差','改善/持平/恶化'],[[r['readout'],r['group'],f(r['raw_edge_absolute_error']),f(r[r['readout']+'_edge_absolute_error']),f(r['mean_difference']),f(r['median_difference']),str(r['improved'])+'/'+str(r['tied'])+'/'+str(r['worsened'])] for r in group_rows]),'',
           '## 结构代价：各seed两图平均后比较','',table(['训练seed','读出','W1变化','联合度TV变化','三角数量误差变化','LCC变化','孤立点变化','W1>10%','LCC下降>19.2'],[[r['train_seed'],r['readout'],f(r['W1_incident4_difference']),f(r['joint_degree_TV_difference']),f(r['triangle_absolute_error_difference']),f(r['LCC_difference']),f(r['isolates_difference']),r['W1_gt_10_percent'],r['LCC_drop_gt_19_2']] for r in pair_rows]),'',
           '全部逐图原值、参考值、绝对误差、分布误差与未定义原因见all_metric_values.csv；逐关系数量、类型三角、度尾部、MP1/MP2/MP3及组成见原始metrics和unified_evaluation.json。W1主口径incident4，all10兼容值为其0.4倍。闭合机会数保留冻结评价定义，PubMed没有IMDB专用导演—电影—演员闭合指标。','',
           '## 比较与证据边界','',
           'EMA0.99为预先指定主要比较，0.9和0.97为平滑尺度敏感性；不得挑最优尺度后改写主假设。八seed是同一PubMed子图上的探索证据；旧2参与过方案选择，新增6单列。每seed两张图和三种EMA读出不是独立训练重复。参数平滑兼有滞后，不能把成功唯一解释为消除末期噪声。预警是人为结构代价提示，未触线不等于安全，也不以某seed触线删除整轮信息。','',
           '若改善只出现在某个尺度或部分seed，按实际范围报告；若边数更准但尾部、三角、关系或连通退化，完整保留取舍。不宣称EMA已修复角色机制或跨图泛化，不启动后续实验。','',
           '## 文件与复算','',
           'make_report.py为本轮CPU表格导出；audit_results.py独立核对全部64张离散图关键指标、逐seed算术、检查点与32读出。独立审计结果另见audit_results.json。训练初态、中点、最终检查点和导出权重保留本地，公开包只登记路径、字节数和哈希，不上传大体积二进制权重；生成原图、日志、指标、配置及源码完整保留。原日志中的运行中progress可能停留running，完成状态以summary/launcher_summary及本次审计为准。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (out/'assessment.json').write_text(json.dumps(assessment,indent=2,allow_nan=False));(out/'source_info.json').write_text(json.dumps(dict(commit='95b1ee86b1d90a7edb102d2d014a024087237316',source=str(src),source_sha256=a['source_sha256'],run=str(run),time=time.time()),indent=2))
    # Graphical overview uses all seeds; no seed/decay selection.
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.8));positions=np.arange(len(c.SEEDS));width=.19
    for i,kind in enumerate(c.READOUTS):
        values=[next(z for z in cell_rows if z['train_seed']==s and z['readout']==kind)['edge_absolute_error'] for s in c.SEEDS]
        axes[0].bar(positions+(i-1.5)*width,values,width,label=kind)
    axes[0].set_xticks(positions);axes[0].set_xticklabels([str(s)[-4:] for s in c.SEEDS]);axes[0].set_ylabel('Mean per-graph |edges - 926|');axes[0].set_xlabel('Training seed (last4 digits; first2 = development)');axes[0].legend(fontsize=8)
    for kind in c.DECAYS:axes[1].plot(positions,[next(z for z in pair_rows if z['train_seed']==s and z['readout']==kind)['edge_error_difference'] for s in c.SEEDS],marker='o',label=kind)
    axes[1].axhline(0,color='black',linewidth=.7);axes[1].set_xticks(positions);axes[1].set_xticklabels([str(s)[-4:] for s in c.SEEDS]);axes[1].set_ylabel('EMA - raw absolute edge error');axes[1].legend();fig.tight_layout();fig.savefig(out/'edge_error_overview.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(12,12))
    for ax,seed in zip(axes.flat,c.SEEDS):
        rows=[r for r in train if r['train_seed']==seed];values=np.array([r['loss'] for r in rows]);smoothed=np.convolve(values,np.ones(50)/50,mode='valid')
        ax.plot(np.arange(50,1001),smoothed,linewidth=1);ax.set_title(str(seed));ax.set_xlabel('normal2 updates');ax.set_ylabel('CE, rolling50 (display only)')
    fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=140);plt.close(fig)
    check();print(json.dumps(dict(graphs=len(graph_rows),cells=len(cell_rows),paired_seed_decay=len(pair_rows),validation_rows=len(validation),status='exported')))
if __name__=='__main__':main()
