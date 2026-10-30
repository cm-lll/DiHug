"""Read-only result audit/export; never imports a model or uses CUDA."""
import csv,json,pathlib,hashlib,time,math,collections,shutil,zipfile,datetime
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
SOURCE=pathlib.Path('/data2/lyh/gnn_project/DiHug/research_branches/hgen_192_full_reference/experiments/hgen_192_full_reference_v1')
PROJECT=pathlib.Path('/data2/lyh/gnn_project/DiHug')
PRIOR=PROJECT/'results/count_aux_results_closeout_20261030_final'
def read(p):return json.loads(pathlib.Path(p).read_text())
def save(p,v):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def csvout(name,rows):
 p=ROOT/'tables'/name;p.parent.mkdir(exist_ok=True)
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def average(values):return sum(values)/len(values) if values and all(v is not None for v in values) else None
def stamp(t):return datetime.datetime.fromtimestamp(t,datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def audit(doc):
 checks=0;failures=[]
 def ck(ok,label):
  nonlocal checks
  checks+=1
  if not ok:failures.append(label)
 manifest=read(ROOT/'source_manifest.json')
 for p,h in manifest.items():ck(sha(SOURCE/p)==h,'source:'+p)
 records={(r['dataset'],r['train_seed'],r['sample_seed'],r['version']):r for r in doc['records']}
 for item in doc['inventory']:
  d,s,ss=(item[k] for k in ('dataset','train_seed','sample_seed'));folder=ROOT/'run'/f'{d}_s{s}'/f'sample_{ss}'
  if not (folder/'binding.json').exists():continue
  b=read(folder/'binding.json');ck(sha(folder/'graph.npz')==b['graph_sha256'],'graph hash');ck(sha(folder/'raw_adjacency.npz')==b['raw_sha256'],'raw hash')
  z=np.load(folder/'graph.npz');a=np.load(folder/'raw_adjacency.npz')['adjacency'];es=z['edges'];types=z['types'];schema=z['schema'];ref=np.load(ROOT/'data'/d/'reference.npz')
  ck(a.shape==(192,192) and np.array_equal(a,a.T),'raw symmetry');ck(np.array_equal(types,ref['types']),'full node types');ck(np.array_equal(schema,ref['schema']),'schema')
  ck(np.array_equal(es,np.stack(np.triu(a,1).nonzero(),axis=0)),'exact upper export');ck(np.all(es[0]<es[1]) and len(set(map(tuple,es.T)))==es.shape[1],'no duplicate off-diagonal export')
  lookup={tuple(pair):i+1 for i,pair in enumerate(schema)}
  ck(z['relations'].tolist()==[lookup[tuple(sorted((int(types[u]),int(types[v]))))] for u,v in es.T],'relation mapping')
  score_file=folder.parent/'scores.npz';actions=[json.loads(line) for line in (folder/'assembly_actions.jsonl').read_text().splitlines()]
  reconstructed=set()
  for row in actions:
   before=len(reconstructed)
   if row['status']=='accepted':
    seq=row['nodes'];ck([int(types[v]) for v in seq]==row['pattern'],'proposal type mapping')
    reconstructed.update(tuple(sorted((u,v))) for u,v in zip(seq[:-1],seq[1:]))
   ck(row['added']==len(reconstructed)-before and row['directed_edges']==2*len(reconstructed),'proposal edge accounting')
  if actions:ck(reconstructed==set(map(tuple,es.T)),'all edges from accepted proposals')
  if b['reason']=='upstream_low_support_return':
   import scipy.sparse as sp
   sc=sp.load_npz(score_file);ck(np.array_equal(a,((sc+sc.T)>0).toarray()),'native low support return')
  if not b['simple']:continue
  rr=records.get((d,s,ss,'original'));ck(rr is not None,'original evaluation exists')
  if rr is None:continue
  degree=a.sum(0);tri=int(np.trace(np.linalg.matrix_power(a.astype(np.int64),3))//6)
  ck(rr['raw']['edges']==es.shape[1] and rr['raw']['nodes']==192,'raw size');ck(rr['raw']['triangles']==tri,'independent triangles');ck(rr['raw']['maximum_degree']==int(max(degree)),'degree maximum');ck(rr['statistics']['total_degree_histogram']==np.bincount(degree.astype(int)).tolist(),'degree histogram')
  ck(rr['statistics']['family_edges']==np.bincount(z['relations']-1,minlength=10).tolist(),'family counts')
  target=ROOT/'derived'/f'{d}_s{s}_{ss}';mapping=read(target/'mapping.json');l=np.load(target/'lcc.npz');ids=l['new_to_original'];le=l['edges']
  ck(np.array_equal(l['types'],types[ids]),'LCC type mapping');ck(set(map(tuple,ids[le].T))<=set(map(tuple,es.T)),'LCC no added edges');ck(sha(target/'lcc.npz')==mapping['derived_sha256'],'LCC binding')
 for r in doc['records']:
  for k,v in r['raw'].items():
   ref=r['reference'].get(k);expected=None if v is None or ref is None else abs(v-ref);actual=r['absolute_error'].get(k)
   ck(actual==expected,'per graph absolute error:'+k)
  a=r['distribution_error']['main_incident4_W1'];b=r['distribution_error']['all10_compat_W1']
  ck(a is None and b is None or a is not None and b is not None and abs(.4*a-b)<1e-10,'W1 incident4/all10')
 for m in doc['cell_means']:
  rr=[r for r in doc['records'] if (r['dataset'],r['train_seed'],r['version'])==(m['dataset'],m['train_seed'],m['version'])]
  vals=[r[m['group']].get(m['metric']) for r in rr]
  expected=average(vals) if len(vals)==2 and all(r['status']=='complete' for r in rr) else None
  ck(m['value']==expected,'strict two-sample means')
 return dict(status='passed' if not failures else 'failed',checks=checks,failures=failures,model_loads=0,new_graphs=0,scope='source and identity hashes, proposal reconstruction, independent size/degree/triangle, LCC and summary arithmetic; not a claim of independent entire metric implementation')
def export(doc):
 core=[];patterns=[];closure=[];spectra=[];retention=[]
 for r in doc['records']:
  ident={k:r[k] for k in ('dataset','train_seed','sample_seed','version','status')};raw,err,dist=(r[k] for k in ('raw','absolute_error','distribution_error'))
  core.append(dict(**ident,edges=raw['edges'],edge_absolute_error=err['edges'],degree_W1=dist['main_incident4_W1'],joint_degree_TV=dist['main_joint_degree_TV'],triangles=raw['triangles'],triangle_absolute_error=err['triangles'],triangle_TV=dist['triangle_composition_TV'],lcc=raw['lcc'],isolates=raw['isolates'],maximum_degree=raw['maximum_degree'],MP2_TV=dist['MP2_composition_TV'],MP3_TV=dist['MP3_composition_TV']))
  st=r['statistics']
  for name,sp in st['typed_degree'].items():
   for rank,degree in enumerate(sp['sorted_degrees'],1):spectra.append(dict(**ident,type=name,rank=rank,degree=degree))
  for kind,key in [('triangle','typed_triangles'),('four_cycle','typed_four_cycles')]:
   counts=st[key];total=sum(counts.values())
   for pattern,value in counts.items():patterns.append(dict(**ident,kind=kind,pattern=pattern,count=value,total=total,composition=value/total if total else None))
  for i,mp in enumerate(st['mp'],1):
   for pattern,value in mp['counts'].items():patterns.append(dict(**ident,kind='MP'+str(i),pattern=pattern,count=value,total=mp['total'],composition=value/mp['total'] if mp['total'] else None))
  for pattern,v in st['typed_twohop_closure'].items():closure.append(dict(ident,pattern=pattern,**v))
 for x in doc['inventory']:
  retention.append({k:x.get(k) for k in ('dataset','train_seed','sample_seed','status','reason','node_retention','edge_retention','type_counts_original','type_counts_lcc')})
 csvout('presentation_core.csv',core);csvout('all_pattern_counts_compositions.csv',patterns);csvout('all_typed_closure.csv',closure);csvout('complete_degree_spectra.csv',spectra);csvout('lcc_retention.csv',retention)
 return core
def report(doc,core):
 lines=['# HGEN完整192节点图测试','',f'科学源码：`{read(ROOT/"source_version.json")["commit"]}`。8个GAN训练，16次预定组装；每数据集4个GAN seed，每模型2图。','',f'实际结果状态：{doc["status"]}；原图记录{sum(r["version"]=="original" for r in doc["records"])}，LCC另列。','', '## 原图主要结果（每seed先平均两图）','', '|数据集|训练seed|边数|边数绝对误差|incident4 W1|联合度TV|三角绝对误差|三角组成TV|LCC|孤立点|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for d,seeds in read(SOURCE/'protocol.json')['train_seeds'].items():
  for s in seeds:
   rr=[x for x in core if (x['dataset'],x['train_seed'],x['version'])==(d,s,'original')]
   def val(k):
    m=average([r[k] for r in rr]) if len(rr)==2 and all(r['status']=='complete' for r in rr) else None
    return 'null' if m is None else f'{m:.5g}'
   lines.append('|'+d+'|'+str(s)+'|'+'|'.join(val(k) for k in ('edges','edge_absolute_error','degree_W1','joint_degree_TV','triangle_absolute_error','triangle_TV','lcc','isolates'))+'|')
 lines+=['','## 比较边界','', '- 训练材料是完整参考图，不留出40%，组装从空邻接开始，无60%参考边骨架。','- HGEN原生组装器接收参考总边数预算；扩散无硬数量配额。边数接近不能单独解释为学习优势。保留原低支持返回、超过目标一条/数条的行为，不修图。','- 采用作者HGEN核心＋完整参考适配＋已声明的零支持整条提案拒绝兼容。正支持夹具与原函数输出/RNG一致，但不是原论文留出表格的原样复现。','- 每数据集只拟合一次专属32维Hin2Vec；4个GAN训练seed共享它。训练路径分别固定抽24000条，5epoch，3750次D与750次G更新；不等于DiHug的normal2更新。','- 作者整数节点抽样的梯度局限保留，未做straight-through修复。已知node head无梯度不是本次擅自修补的内容。','- 与旧CE／CE_COUNT的3000／6000结果仅作相同参考下的描述性对照，不宣称相同训练预算或跨方法随机轨迹配对。全指标在cross_method_training_seed_means.csv逐seed保留。','- 原图为主，LCC是后处理。零孤立点/LCC全连通若由筛选保证，不算学习收益。','- 公共冻结评价器635原值、635绝对误差、73分布项及全部嵌套明细保留。native powerlaw未执行、社区标签与orbit等未实现项保持null；不声称原生HGEN全部评价均已运行。','- 4个训练seed与每模型2图只提供当前两张参考图上的探索性证据；编号边差不代表同构不变多样性。','', '## 文件','', '`evaluation.json`、`tables/per_graph_all_scalar_metrics.csv`、`tables/per_graph_all_detail_fields.csv`保留完整数值；`tables/training_seed_means.csv`、`tables/HGEN_four_seed_summary.csv`保留严格完整性均值；`tables/sample_diversity.json`含样本差异。`independent_audit.json`记录本轮CPU核查。原始图、raw邻接、逐条组装记录及LCC映射均保留；权重留本地，以文件哈希登记，不夹入发布包。']
 lines+=['','## 已确认结论与适用范围','', '1. 本固定配方完整跑完8次GAN/16次从零组装，未补抽；全图条件替代了历史40%留出，但不等于每份随机路径银行必然逐边覆盖。`path_bank_coverage.json`显示IMDB首seed覆盖322/323边，其余三个323/323；PubMed四seed覆盖924/924/924/925条（参考926），这些实际遗漏保留、不补齐。','2. IMDB的三角组成TV在四seed均为1，且平均孤立点40.5至84个，不能因边数相近认为异质结构已保留。PubMed四seed度W1为1.085至1.211，三角数远低于参考1937。固定配方下结构差距在各seed重复出现；这不是对HGEN所有训练预算与变体的普遍否定。','3. HGEN未强制采用扩散的相同关系支持。IMDB的参考零频关系中出现DG/AG/MM/GG等边（具体数量见reference_zero_relations.json）；不能仅凭参考频数0宣称这些关系在数学schema中非法，也不能用这批结果宣称在完全同支持空间下公平击败HGEN。','4. 没有改原GAN整数抽样与梯度规则，没有搜索epoch/路径数或按结果优化HGEN；5epoch是本轮固定测试预算，未证明GAN已充分收敛。下一决定应先审阅基线定义及条件差异，不自动追加模型、训练或生成。','']
 if (ROOT/'native_HGEN_powerlaw_supplement.json').exists():
  extra=read(ROOT/'native_HGEN_powerlaw_supplement.json')
  lines+=['## 原生powerlaw独立补充','',f'公共冻结评价中的HGEN_native_powerlaw仍为null，保持与旧表一致。另利用已安装powerlaw={extra["package_version"]}，原样执行作者statistics_power_law_alpha，单列{len(extra["rows"])}项（两参考＋32原图/LCC），见native_HGEN_powerlaw_supplement.json。xmin=max(min(degrees),1)，不替换公共表中的fixed-xmin1项目或旧值；警告、失败与来源单列。未安装依赖，未引入新指标定义。']
 (ROOT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
def main():
 start=time.time();doc=read(ROOT/'evaluation.json');result=audit(doc);save(ROOT/'independent_audit.json',result);core=export(doc);report(doc,core)
 timings=[]
 for p in sorted((ROOT/'run').glob('*/summary.json')):
  v=read(p);t=read(p.parent/'training_complete.json');w=read(p.parent/'walk_receipt.json')
  timings.append(dict(dataset=v['dataset'],train_seed=v['train_seed'],D_updates=v['D_updates'],G_updates=v['G_updates'],training_seconds=t['seconds'],walk_seconds=w['seconds'],assembly_seconds=sum(x['seconds'] for x in v['requests']),total_seconds=v['seconds'],peak_allocated_mib=v['peak_allocated_mib'],peak_RSS_mib=v['peak_RSS_mib']))
 csvout('timing.csv',timings)
 save(ROOT/'cpu_finalize_receipt.json',dict(epoch=time.time(),Asia_Shanghai=stamp(time.time()),seconds=time.time()-start,audit=result['status'],model_forwards=0,optimizer_updates=0,new_graphs=0))
 print(result)
if __name__=='__main__':main()
