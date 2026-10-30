import pathlib,json,time,hashlib,zipfile,shutil
OUT=pathlib.Path(__file__).parent
def main():
 r=json.loads((OUT/'evaluation.json').read_text());done=json.loads((OUT/'report_completion.json').read_text());ledger=json.loads((OUT/'ledger.json').read_text());ref=json.loads((OUT/'run/derived_lcc_evaluation/IMDB_reference.json').read_text())
 fields=[('absolute_error','edges','边数绝对误差'),('distribution_error','main_incident4_W1','incident4度W1'),('distribution_error','main_joint_degree_TV','联合度TV'),('absolute_error','triangles','三角数绝对误差'),('distribution_error','triangle_composition_TV','三角组成TV'),('absolute_error','lcc','LCC节点数误差'),('absolute_error','type_Genre_maximum','Genre最大度误差'),('absolute_error','type_Genre_second','Genre次大度误差'),('absolute_error','type_Genre_low_le1_fraction','Genre低度比例误差'),('absolute_error','triangle_(0, 1, 2)','DAM数量误差'),('absolute_error','triangle_(0, 1, 1)','DAA数量误差'),('absolute_error','triangle_(1, 1, 2)','AAM数量误差'),('distribution_error','MP2_composition_TV','MP2组成TV'),('distribution_error','MP3_composition_TV','MP3组成TV')]
 idx={(x['train_seed'],x['arm'],x['checkpoint'],x['version'],x['group'],x['metric']):x['value'] for x in r['cell_means']}
 pairs={(x['train_seed'],x['checkpoint'],x['version'],x['group'],x['metric']):x['difference'] for x in r['paired_means']};f=lambda v:'null/未完成' if v is None else f'{v:.6g}'
 text=['# IMDB同轨迹3000→6000与数量辅助项比较','',f"6000步新图完成{done['new_complete_graphs']}/8；原3000步8图只读复用。原图为主，LCC单列；全部使用对应EMA0.99。",'', '每张图先取固定参考误差，再平均两个采样seed；数量原值差、误差差和分布距离差分开。训练seed分别展示，缺图/未定义不以0代替。','']
 for s in [2026102911,2026102931]:
  text += [f'## 训练seed {s}','', '|指标（误差越小越好）|CE3000|CE6000|COUNT3000|COUNT6000|6000 COUNT−CE|','|---|---:|---:|---:|---:|---:|']
  for g,k,label in fields:
   vals=[idx.get((s,a,u,'original',g,k)) for a,u in [('CE',3000),('CE',6000),('CE_COUNT',3000),('CE_COUNT',6000)]]
   text.append('|'+label+'|'+'|'.join(f(v) for v in vals)+'|'+f(pairs.get((s,6000,'original',g,k)))+'|')
  text.append('')
 text += ['## 完整指标与边界','',
  '- 原图/LCC的全部635原值、对应逐图绝对误差、73分布指标及全部变长结构字段见evaluation.json与tables。含各关系、完整度谱/联合度、类别三角、闭合分子/机会/比例、真四环/legacy K4、MP1/2/3与walk、距离/连通、样本差异。',
  '- METRIC_DIRECTORY.md及metric_catalog.csv逐字段注明定义、单位、参考和方向；HGEN原生powerlaw/社区标注/orbit等缺项不补造数字。附带历史metric_definition_reference.md只说明实现来源，其中旧PubMed实测不是当前IMDB参考。',
  '- 本票参考是IMDB192节点323边、分量188/2/2。LCC派生指标沿用此前固定参考合同，不偷偷把参考也裁成LCC；派生图自身连通和无孤立点不算模型能力。',
  '- 不能只凭边数更准宣布改善；Genre排序度谱、DAM/DAA/AAM、闭合机会与比例和MP同时检查。完整原始数量差和误差差保留，表中只作重点摘要。',
  '- 两个训练seed、每格两图仍是探索。更长训练不自动等于收敛；收益混合或方向翻转均保留，不选最好检查点、不调整公式或lambda。',
  '- 原PubMed主任务与截止未修改；本票08:30科学停止、09:00交付，不自动再续训或补抽。','']
 (OUT/'RESULTS.md').write_text('\n'.join(text));status='complete' if done['new_complete_graphs']==8 and not done['errors'] else 'incomplete'
 (OUT/'completion.json').write_text(json.dumps(dict(status=status,finished_epoch=time.time(),elapsed_minutes=(time.time()-ledger['start_epoch'])/60,new_updates=sum(json.loads(p.read_text()).get('new_updates',0) for p in (OUT/'run').glob('IMDB*/summary.json')),new_graphs=done['new_complete_graphs'],graph_counts=done['graph_counts'],errors=done['errors'],audit=json.loads((OUT/'independent_audit.json').read_text())['status'],parent_untouched=True),indent=2))
 source=pathlib.Path('/data2/lyh/gnn_project/DiHug/research_branches/imdb_count_aux_u6000/experiments/imdb_count_aux_continue_v1')
 shutil.copytree(source,OUT/'adapter_source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix not in ('.pt','.zip','.tmp') and p.name not in ['payload_manifest.json','ZIP_SHA256.txt'] and 'tmp' not in p.relative_to(OUT).parts]
 manifest={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files};(OUT/'payload_manifest.json').write_text(json.dumps(manifest,indent=2))
 z=OUT/'IMDB_3000_to_6000_full_evaluation.zip'
 with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
  for p in files+[OUT/'payload_manifest.json']:f.write(p,str(p.relative_to(OUT)))
 (OUT/'ZIP_SHA256.txt').write_text(hashlib.sha256(z.read_bytes()).hexdigest()+'  '+z.name+'\n')
 print((OUT/'completion.json').read_text())
if __name__=='__main__':main()
