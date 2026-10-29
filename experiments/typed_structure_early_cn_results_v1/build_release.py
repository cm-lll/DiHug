"""Archive only completed evidence; no model execution or GPU access."""
import csv,datetime,hashlib,json,pathlib,shutil,time,zipfile
P=pathlib.Path(__file__).resolve().parent
ROOT=P.parents[1]
SRC=ROOT/'research_branches/typed_structure_final_screen_review/experiments/typed_structure_early_cn_v1'
LIB=SRC.parent/'typed_structure_final_screen_results_v1/audit.py'
OUT=P/'release_payload';RUN=P/'run'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    OUT.mkdir(exist_ok=False)
    for f in RUN.rglob('*'):
        if f.is_file() and f.suffix!='.pt' and 'tmp' not in f.relative_to(RUN).parts:
            q=OUT/'run'/f.relative_to(RUN);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,q)
    for name in ['approval.json','startup_receipt.json','prelaunch_binding.json','resource_precheck.json','resource_precheck.csv','process_precheck.csv','launcher_process.json','launcher.log','independent_audit.json','audit_completed.py','build_release.py']:
        f=P/name
        if f.exists():shutil.copy2(f,OUT/name)
    shutil.copy2(LIB,OUT/'audit_library.py')
    with zipfile.ZipFile(OUT/'runtime_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(SRC.rglob('*')):
            if f.is_file() and not any(x in f.relative_to(SRC).parts for x in ['__pycache__','.pytest_cache','evidence']):z.write(f,'typed_structure_early_cn_v1/'+str(f.relative_to(SRC)))
    ev=read(RUN/'unified_evaluation.json');audit=read(P/'independent_audit.json')
    for name in ['cell_means','paired','paired_means','lcc_minus_original']:
        rows=ev[name]
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    rows=[]
    for r in ev['records']:
        for group in ['raw','reference','absolute_error','distribution_error']:
            for metric,value in r[group].items():rows.append(dict(dataset=r['dataset'],train_seed=r['train_seed'],arm=r['arm'],sample_seed=r['sample_seed'],version=r['version'],group=group,metric=metric,value=value))
    with (OUT/'per_graph_long.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def mean(d,a,g,k,v='original'):
        vals=[r[g].get(k) for r in ev['records'] if r['dataset']==d and r['arm']==a and r['version']==v]
        assert len(vals)==2
        return None if None in vals else sum(vals)/2
    lines=['# E-earlyCN：原训练seed两任务结果','',
        '固定科学源码 dbe96f519443e8f8937d43cd191803658bf83464；CN37→64零初始化无偏置投影进入主干首层前边表示，末端CN37列移除；保留ReFeX与冻结RolX，无Dropout。',
        '', '北京时间11:46:47.481317启动，15:05:41.398028完成训练、4条T1000链和自动评价；原科学停止15:46:47.481317、交付截止16:06:47.481317。2×3000次normal2、EMA0.99、采样seed2026102821/2026102822全部完成，无失败、补抽或重启。科学GPU0/1已释放。',
        '', '**初步结论：本seed的IMDB数量、度和连通改善，类型三角总量没有改善；PubMed未形成统一收益，两次采样分化明显。不能据此采用统一早融合方案。** 新seed与D早融合结果由独立六任务批次交付；这里不提前并入未完成结果。',
        '', '|数据集|组别|边数均值|逐图绝对边误差均值|incident4 W1|联合度TV|三角误差均值|三角组成TV|LCC均值|', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for d,arms in [('IMDB',['E','D','E_earlyCN']),('PubMed',['E','C','E_earlyCN'])]:
        for a in arms:
            vals=[mean(d,a,'raw','edges'),mean(d,a,'absolute_error','edges'),mean(d,a,'distribution_error','main_incident4_W1'),mean(d,a,'distribution_error','main_joint_degree_TV'),mean(d,a,'absolute_error','triangles'),mean(d,a,'distribution_error','triangle_composition_TV'),mean(d,a,'raw','lcc')]
            lines.append('|'+d+'|'+a+'|'+'|'.join('null' if v is None else f'{v:.6g}' for v in vals)+'|')
    lines+=['','IMDB两图327/316边（参考323），W1为0.080729/0.070313，三角均43（参考45），LCC186/188；旧E为304/309边、LCC183/183。相对旧E平均绝对边误差16.5→5.5、W1 0.085938→0.075521，但类型三角及闭合应按完整表解释，不能只看总三角。',
        '', 'PubMed两图949/696边（参考926），W1为0.559896/0.723958，三角3372/1406（参考1937）；旧E为823/786边。平均绝对边误差121.5→126.5，W1 0.588542→0.641927，三角绝对误差167→983；Chemical低度≤1节点均16，旧E均20，参考值见逐图reference字段。低度减少并未同时带来整体结构改善。旧C边数919/859，仍是较好参照。',
        '', '## 原图与LCC', '', '|数据集|采样seed|原节点/边|LCC节点/边|类型损失|', '|---|---|---|---|---|']
    for x in ev['inventory']:
        j=x['job'];rs=[r for r in ev['records'] if r['dataset']==j['dataset'] and r['arm']=='E_earlyCN' and r['sample_seed']==j['sample_seed']]
        rr={r['version']:r for r in rs}
        lines.append(f"|{j['dataset']}|{j['sample_seed']}|{rr['original']['raw']['nodes']}/{rr['original']['raw']['edges']}|{rr['lcc']['raw']['nodes']}/{rr['lcc']['raw']['edges']}|{x['type_nodes_lost']}|")
    lines+=['','LCC为后处理使用版，单独评价；保留全部内部原边及节点映射，无补边、删异常点或改变采样。LCC自身连通不是模型额外收益。原图是主要结论对象。',
        '', '## 证据与局限', '', f"独立CPU审计通过：4原图/4LCC、12份权重文件哈希及最终EMA导出、6000次normal2记录、4000步覆盖、固定验证输入、{audit['numeric_aggregate_checks']}项汇总算术。独立重算边/关系/度、类别三角、真正四环、MP1/2/3总量、聚类、连通、incident4/all10 W1、联合TV和LCC诱导映射；MP组成和闭合细分继承冻结评价器，本次未单独枚举复算。",
        '', 'per_graph_long.csv保留全部原值、固定参考、逐图绝对误差与分布误差；paired.csv/paired_means.csv保留新E−旧E及IMDB D/PubMed C逐采样seed与均值，null不丢弃。incident4为主、all10兼容值为其0.4倍。HGEN legacy K4/聚类与真正四环/标准聚类分别保留；原生powerlaw依赖、社区标签、orbit等未实现项沿用原缺项，不称完整原生HGEN。',
        '', '每数据集仅1个训练seed、2个生成样本；这是开发比较，不能推断跨seed稳定性、收敛或完整图有效。两个样本的边身份差异不是充分拓扑多样性证据。跨后端数值等价历史失败不改写。本次审计不构建模型，不新增前向/训练/生成。',
        '', '大权重留在原results目录，不上传；independent_audit.json登记完整路径、字节数、SHA256。runtime_source.zip含冻结源码、数据、字典、配置与指标定义；旧控制来源和哈希在baseline_bindings.json。运行审批是已结束的原批次证据，不授权新任务。',
        '', 'CPU独立复核：解包runtime_source.zip至独立数据盘目录，再执行 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python audit_completed.py --run run --source <解包目录>/typed_structure_early_cn_v1 --audit-library audit_library.py --output <新的审计JSON> --public-only`。需要numpy/networkx；无需Torch/GPU。','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    (OUT/'weight_index.json').write_text(json.dumps(audit['weights'],indent=2))
    (OUT/'release_scope.json').write_text(json.dumps(dict(source_commit='dbe96f519443e8f8937d43cd191803658bf83464',source_sha256=audit['source_sha256'],new_models=2,new_updates=6000,new_chains=4,graph_versions=8,old_control_records=16,independent_audit_passed=True,scientific_complete_epoch=audit['scientific_complete_epoch'],build_epoch=time.time(),limits=audit['limits']),indent=2))
    index={str(f.relative_to(OUT)):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(OUT.rglob('*')) if f.is_file()}
    (OUT/'PAYLOAD_SHA256.json').write_text(json.dumps(index,indent=2))
    (OUT/'SHA256SUMS').write_text(''.join(v['sha256']+'  '+k+'\n' for k,v in index.items()))
    print(json.dumps(dict(output=str(OUT),files=len(index),bytes=sum(v['bytes'] for v in index.values()))))
if __name__=='__main__':main()
