"""Build final six-task evidence and cross-batch report, CPU only."""
import csv,datetime,hashlib,json,os,pathlib,shutil,subprocess,sys,time,zipfile
P=pathlib.Path(__file__).resolve().parent;ROOT=P.parents[1]
SRC=ROOT/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1'
LIB=ROOT/'research_branches/typed_structure_final_screen_review/experiments/typed_structure_final_screen_results_v1/audit.py'
OUT=P/'release_payload_final';RUN=P/'run_retry1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 start=time.time();OUT.mkdir(exist_ok=False)
 for f in RUN.rglob('*'):
  if f.is_file() and f.suffix!='.pt' and not any(k in f.relative_to(RUN).parts for k in ['tmp','__pycache__']):
   dest=OUT/'run'/f.relative_to(RUN);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
 names=['approval.json','approval_retry1.json','approval_retry1_diff.json','retry_environment_actual.json','retry_preflight.json','startup_receipt.json','startup_receipt_retry1.json','formal_retry_once.py','retry_once_claim.json','retry_launch_process.json','approved_cost.json','binding_receipt.json','independent_audit_final.json','audit_final.py','build_final_release.py','resources_before.csv','processes_before.csv','retry_launcher.log','launcher.log']
 for name in names:
  f=P/name
  if f.exists():shutil.copy2(f,OUT/name)
 first=OUT/'first_failed_attempt';first.mkdir()
 for f in (P/'run').rglob('*.json'):
  if f.name in ['activation.json','launcher_summary.json','summary.json']:
   dest=first/f.relative_to(P/'run');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
 shutil.copy2(LIB,OUT/'audit_library.py')
 with zipfile.ZipFile(OUT/'runtime_source.zip','w',zipfile.ZIP_DEFLATED) as z:
  for f in sorted(SRC.rglob('*')):
   if f.is_file() and not any(k in f.relative_to(SRC).parts for k in ['__pycache__','.pytest_cache','evidence']):z.write(f,'typed_structure_early_cn_expansion_v1/'+str(f.relative_to(SRC)))
 ev=read(RUN/'unified_evaluation.json');audit=read(P/'independent_audit_final.json')
 for name in ['cell_means','paired','paired_means','lcc_minus_original']:
  rows=ev[name]
  with (OUT/(name+'.csv')).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 rows=[];detail=[]
 for r in ev['records']:
  ident={k:r[k] for k in ['dataset','train_seed','arm','sample_seed','version']}
  for group in ['raw','reference','absolute_error','distribution_error']:
   for metric,value in r[group].items():rows.append(dict(**ident,group=group,metric=metric,value=value))
  detail.append(dict(**ident,typed_triangles=json.dumps(r['statistics']['typed_triangles']),family_edges=json.dumps(r['statistics']['family_edges']),typed_sorted_degrees=json.dumps({k:v['sorted_degrees'] for k,v in r['statistics']['typed_degree'].items()}),closure=json.dumps(r['statistics'].get('closure',r['statistics'].get('typed_closure',{}))),mp_totals=json.dumps([v['total'] for v in r['statistics']['mp']])))
 for name,rs in [('per_graph_long',rows),('structure_details',detail)]:
  with (OUT/(name+'.csv')).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
 keys=[('raw','edges'),('absolute_error','edges'),('distribution_error','main_incident4_W1'),('distribution_error','main_joint_degree_TV'),('absolute_error','triangles'),('distribution_error','triangle_composition_TV'),('raw','lcc')]
 def mean(d,s,a,g,k,v='original'):
  rs=[r[g].get(k) for r in ev['records'] if (r['dataset'],r['train_seed'],r['arm'],r['version'])==(d,s,a,v)];assert len(rs)==2
  return None if None in rs else sum(rs)/2
 def table(d,s,arms):
  out=['|训练seed|组别|边数均值|绝对边误差|incident4 W1|联合度TV|绝对三角误差|三角组成TV|LCC|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
  for arm in arms:
   vals=[mean(d,s,arm,g,k) for g,k in keys];out.append('|'+str(s)+'|'+arm+'|'+'|'.join('null' if v is None else f'{v:.6g}' for v in vals)+'|')
  return out
 lines=['# CN早融合：新增六任务及跨批次最终比较','',
 '**结论：IMDB的E早融合在两个训练seed上均改善数量误差、度W1和连通；PubMed数量收益随训练seed翻转，度W1在两seed上均恶化。当前不能把早融合当作统一修复。** IMDB新seed也存在三角总量误差变大而组成TV变好的取舍。',
 '', '新增科学源码固定89320ada4d58b133884f62360e9662e713d872f2；原两任务固定dbe96f519443e8f8937d43cd191803658bf83464。旧晚融合与PubMed C来自aa4227b9727ca6853ba7e26412ce6d438a0fa971。全部同192节点、3000 normal2、EMA0.99、两固定生成seed，原图为主、LCC单列。未改变模型、字典、噪声、损失、查询、后验或科学设置；无Dropout、门控、概率校准和修图。',
 '', '## 完成范围与时钟', '',
 '新增6模型×3000=18000更新、12条T1000完整链和自动评价在北京时间16:10:54.304137完成，无科学运行失败、缺图、补抽或自动重启；GPU2—7释放。原起点12:41:47.946999保持，科学停止17:01:47.946999、最终截止17:21:47.946999未重置。',
 '', '首次启动审批漏environment，六worker在模型构造之前退出，0更新/0链；旧审批和日志保留。用户授权唯一一次重试，先无模型核验实际environment与全部启动字段，12:49:20.097300执行重试，继承原时钟，全部完成。修复占用原故障余量。实际发布源码绑定较准备摘要多PAYLOAD_SHA256.json，按实际源码绑定803c4e933b95cb547e2d21dd2fd0570db6623d032c35462d14984a1e29518b7d如实记录；不是首次worker失败原因。',
 '', '原两任务在15:05:41.398028完成2模型6000更新、4图及自动评价，15:46科学停止/16:06最终截止未改。本地报告/ZIP在原截止前复验；原独立GitHub发布被自动审批服务超时阻塞，本批只读归入其封存结果，不宣称原批按时远端发布。总体8模型24000更新、16图，两个批次时钟分别保留。',
 '', '## IMDB：逐seed早晚融合', '']
 lines+=table('IMDB',2026102911,['D_lateCN','D_earlyCN','E_lateCN','E_earlyCN'])+['']+table('IMDB',2026102931,['E_lateCN','E_earlyCN'])
 lines+=['', 'E早对E晚：原seed边误差16.5→5.5、W1 0.085938→0.075521、LCC183→187；新seed边误差7.5→0.5、W1 0.089844→0.085938、LCC181→186。两seed方向一致，但W1增益较小、样本很少。',
 '', '三角必须分解：新seed早融合总三角40/43，晚融合45/44，参考45；总量误差0.5→3.5。早融合DAA/AAM均0，DAM为40/43；晚融合DAA均1、AAM为4/2、DAM40/41。总三角更接近不等于类型闭合更好。原seed早融合DAM41/43、DAA1/0、AAM1/0。完整机会数、闭合分子/分母/比率和Genre排序度谱见逐图表及structure_details.csv。',
 '', '原seed D早对D晚：边误差10→7，W1 0.095052→0.096354略差；三角误差1.5→2.5。E早对D早：边误差7→5.5、W1 0.096354→0.075521，联合度TV也改善；仅一个训练seed，不宣称RolX普遍有效。',
 '', '## PubMed：逐seed早晚融合','']
 lines+=table('PubMed',2026102912,['C','D_lateCN','D_earlyCN','E_lateCN','E_earlyCN'])+['']+table('PubMed',2026102932,['E_lateCN','E_earlyCN'])
 lines+=['', 'E早对E晚：原seed边误差121.5→126.5、W1 0.588542→0.641927、三角误差167→983，未改善；新seed边误差76.5→50.5，但W1 0.476563→0.498698恶化。新seed三角数量误差303.5→281略降，组成TV约0.0700→0.1366明显增加。不能用新seed数量收益掩盖原seed翻转或类型结构代价。',
 '', 'Chemical低度≤1节点：原seed晚20/20、早16/16；新seed晚14/9、早12/10。新seedChemical最大度晚95/103、早99/107，低度减少同时高尾可更突出。GD/DC逐关系及参考误差完整列出，不把全图数量变化当成所有关系一致改善。',
 '', '原seed D早对D晚：边误差108→169，W1 0.542969→0.597656，三角数量误差903→501；E早对D早：边误差169→126.5，但W1 0.597656→0.641927、三角误差501→983变差。RolX早融合增量同样混合。原C只用于同原训练seed匹配参照，新seed没有C对照，不作跨seed严格配对。',
 '', '## LCC与样本差异', '',
 '每一原图、LCC、原图→LCC变化均保留；LCC按实际节点/类型集合和关系容量计算，incident4使用固定参考类型权重，all10另列0.4倍兼容值。自动移除分量是后处理，不计为模型学会连通。节点/边/逐类型保留比例与映射见run/derived_lcc_evaluation及unified_evaluation.inventory。缺失类型、空分母、未实现项保留null，不按成功字段交集删项。',
 '', '同模型两样本的边身份差、边Jaccard、度W1、三角/MP组成差保留于diversity。每模型仅两样本，编号对齐的边差异不是同构不变的充分多样性保证。没有筛图、去重后重算或补抽。',
 '', '## 审计、成本与可复核范围', '',
 f"独立CPU审计通过：36份新科学权重文件哈希与final/readout一致、18000次normal2记录、12000步完整查询与支持、12原图/12LCC诱导映射、{audit['numeric_aggregate_checks']}项汇总算术。实际早晚theta0、D原初态迁移、3000更新输入流和采样初态/RNG绑定均核对，原两任务只读取证。未构建模型、未新增前向/训练/生成。",
 '', '独立重算边数、关系数、度、类别三角、真正四环、MP1/2/3总量、聚类、连通、固定参考W1、联合TV与三角组成TV。MP/闭合组成细分继承冻结评价器，本次未再独立枚举。所有完整原始指标/参考/逐图误差/配对误差差保留；旧HGEN K4与真正四环、旧聚类与标准聚类分名，原生HGEN缺包/社区标签/orbit等缺项仍披露。',
 '', '|数据集|组别|训练seed|训练分钟|峰allocated MiB|峰reserved MiB|峰RSS MiB|','|---|---|---|---:|---:|---:|---:|']
 for row in audit['training']:
  j=row['job'];lines.append(f"|{j['dataset']}|{j['arm']}|{j['train_seed']}|{row['seconds']/60:.3f}|{row['peak_allocated_mib']:.3f}|{row['peak_reserved_mib']:.3f}|{row['peak_rss_mib']:.3f}|")
 lines+=['', '峰值是各进程独立峰值，不相加作为同时峰值；每条链实际秒数见independent_audit_final.graphs。新增从原激活到自动评价共209.106分钟，包括首次失败和修复等待；归档、审计、发布另有实际回执，不倒改科学完成时间。CPU线程1，无DDP。科学权重/优化器/RNG及保存状态留在原results目录，weight_index.json提供路径、字节及SHA256，不上传大体积权重。',
 '', 'CPU公开复核：在独立数据盘解包runtime_source.zip，运行 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python audit_final.py --run run --source <解包目录>/typed_structure_early_cn_expansion_v1 --library audit_library.py --output <新JSON路径> --public-only`。无需模型或GPU；公开模式重算新图和全部汇总算术，不要求私有权重和历史绝对路径，外部来源由服务器私有审计及其封存回执证明。',
 '', '## 停止与结论边界', '',
 '早融合保持为一个有局部价值但尚未达到统一修复目标的候选，不把成功运行等同效果成功。IMDB两seed数量/W1收益值得记录；PubMed结果混合，不能声称已解决欠边、联合度、尾部和类型闭合。此为开发比较：E早晚仅两个训练seed，D早/RolX早增量仅一个seed；不作显著性、跨图/完整图泛化或收敛保证。',
 '', '固定科学任务已完成，资源已释放。停止，不自动追加训练、生成、Dropout、角色搜索或完整图实验。','']
 (OUT/'RESULTS.md').write_text('\n'.join(lines))
 (OUT/'weight_index.json').write_text(json.dumps(audit['weights'],indent=2))
 (OUT/'release_scope.json').write_text(json.dumps(dict(source_commit='89320ada4d58b133884f62360e9662e713d872f2',source_sha256=audit['source_sha256'],models=6,updates=18000,chains=12,combined_models=8,combined_chains=16,scientific_complete_epoch=audit['scientific_complete_epoch'],build_epoch=time.time(),original_two_source='dbe96f519443e8f8937d43cd191803658bf83464',old_result_commit='aa4227b9727ca6853ba7e26412ce6d438a0fa971',independent_cpu_audit_passed=True),indent=2))
 os.environ['MPLCONFIGDIR']=str(P/'plot_config');os.environ['TMPDIR']=str(P/'plot_tmp');pathlib.Path(os.environ['TMPDIR']).mkdir(exist_ok=True)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 curves=OUT/'curves';curves.mkdir()
 for row in audit['training']:
  j=row['job'];folder=OUT/'run'/j['directory'];rs=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()];vals=read(folder/'validation.json');loss=np.array([r['loss'] for r in rs]);fig,ax=plt.subplots(1,2,figsize=(10,3.5))
  ax[0].plot(range(1,len(loss)+1),loss,alpha=.2,linewidth=.4);ax[0].plot(range(50,len(loss)+1),np.convolve(loss,np.ones(50)/50,'valid'),label='50-update mean');ax[0].legend(fontsize=6)
  for kind in ['raw','ema_0.99']:
   for t in [1,500,1000]:ax[1].plot([x['updates'] for x in vals],[next(r['ce_sum_over_S'] for r in x[kind]['rows'] if r['t']==t) for x in vals],marker='o',label=f'{kind} t={t}')
  ax[1].legend(fontsize=6)
  for a in ax:a.set_xlabel('normal2 updates');a.set_ylabel('CE / full schema S')
  fig.suptitle(j['id']);fig.tight_layout();fig.savefig(curves/(j['id']+'.png'),dpi=140);plt.close(fig)
 index={str(f.relative_to(OUT)):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(OUT.rglob('*')) if f.is_file()}
 (OUT/'PAYLOAD_SHA256.json').write_text(json.dumps(index,indent=2));(OUT/'SHA256SUMS').write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in index.items()))
 archive=P/'typed_structure_early_cn_expansion_results_v1.zip'
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for f in sorted(OUT.rglob('*')):
   if f.is_file():z.write(f,str(f.relative_to(OUT)))
 scratch=P/'final_unpack_check';scratch.mkdir(exist_ok=False)
 with zipfile.ZipFile(archive) as z:
  for m in z.infolist():assert scratch.resolve() in (scratch/m.filename).resolve().parents and (m.external_attr>>16)&0o170000!=0o120000
  z.extractall(scratch)
 for n,b in index.items():assert (scratch/n).stat().st_size==b['bytes'] and sha(scratch/n)==b['sha256'],n
 unpack=scratch/'source_unpacked';unpack.mkdir()
 with zipfile.ZipFile(scratch/'runtime_source.zip') as z:
  for m in z.infolist():assert unpack.resolve() in (unpack/m.filename).resolve().parents
  z.extractall(unpack)
 subprocess.run([sys.executable,str(scratch/'audit_final.py'),'--run',str(scratch/'run'),'--source',str(unpack/'typed_structure_early_cn_expansion_v1'),'--library',str(scratch/'audit_library.py'),'--output',str(scratch/'public_audit.json'),'--public-only'],check=True,timeout=180)
 receipt=dict(status='passed',finished_epoch=time.time(),seconds=time.time()-start,payload_files=len(index),zip_sha256=sha(archive),zip_bytes=archive.stat().st_size,zip_path=str(archive),independent_extraction=True,public_audit=read(scratch/'public_audit.json'))
 (P/'archive_verification_final.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:receipt[k] for k in ['status','seconds','payload_files','zip_sha256','zip_bytes']}))
if __name__=='__main__':main()
