"""Package completed evidence, excluding model/optimizer weights and credentials."""
import pathlib,shutil,json,hashlib,zipfile,time,subprocess
ROOT=pathlib.Path(__file__).resolve().parent
PROJECT=pathlib.Path('/data2/lyh/gnn_project/DiHug')
SOURCE=PROJECT/'research_branches/hgen_192_full_reference/experiments/hgen_192_full_reference_v1'
PRIOR=PROJECT/'results/count_aux_results_closeout_20261030_final'
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def main():
 started=time.time();dest=ROOT/'release_payload';dest.mkdir(exist_ok=False);weights=[]
 def cp(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 for folder in ('run','derived','data','engineering'):
  for p in (ROOT/folder).rglob('*'):
   if not p.is_file() or '__pycache__' in p.parts:continue
   if p.suffix=='.pt':weights.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size));continue
   cp(p,dest/p.relative_to(ROOT))
 for p in ROOT.iterdir():
  if p.is_file() and p.suffix in ('.json','.md','.py','.png','.pdf','.log','.patch') and p.name not in ('package_receipt.json','archive_audit.json','publication.json'):
   cp(p,dest/p.name)
 for p in (ROOT/'tables').rglob('*'):
  if p.is_file():cp(p,dest/p.relative_to(ROOT))
 for p in SOURCE.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:cp(p,dest/'source'/p.relative_to(SOURCE))
 for p in (PRIOR/'frozen_evaluation_source').rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:cp(p,dest/'frozen_evaluation_source'/p.relative_to(PRIOR/'frozen_evaluation_source'))
 for d in ('IMDB','PubMed'):cp(ROOT/'data'/d/'reference.npz',dest/'reference'/(d+'.npz'))
 cp(PRIOR/'metric_definition_reference.md',dest/'metric_definition_reference.md')
 save(dest/'local_weights_index.json',dict(files=weights,uploaded=False,note='scientific and engineering weights explicitly identified; engineering states never entered scientific fits'))
 patch=subprocess.check_output(['git','diff','b656f75c412b8fa067f800a00eb886cab924486d','bfbd7b4b864b354fb2dbcb82d1056e3297fc7904','--','experiments/hgen_192_full_reference_v1'],cwd=str(SOURCE.parents[1]))
 (dest/'source_minimal.patch').write_bytes(patch)
 (dest/'README.md').write_text('HGEN全192节点参考图基线：先读REPORT_ZH.md和METRIC_DIRECTORY.md。科学源码source/逐字节绑定bfbd7b4；结果与CPU派生脚本单列，不改变冻结科学实现。\n\n8个GAN拟合/16原图/16LCC已完成。原图run/*/sample_*/graph.npz；其edges为2×E、relations为1-based，data/*/reference.npz的edges为E×2。raw_adjacency和全部组装动作保留。完整指标evaluation.json与tables/。HGEN接收总边预算，四GAN seed共享每图一次Hin2Vec，原生梯度限制与零支持兼容见报告；不是原论文40%留出原样复现。\n\n原科学运行固定本地路径，不要直接启动source/launch.py重跑。可独立复算CPU评价：OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B recompute_cpu.py --root . --output NEW_CPU_OUTPUT。使用现有numpy/scipy/networkx，不加载模型。模型/优化器.pt文件未打包；本地路径及哈希见local_weights_index.json。\n\nSHA256清单见payload_manifest.json；ZIP外部哈希另附。没有凭据或无关项目文件。原始记录不改，未自动追加候选或样本。\n')
 manifest={str(p.relative_to(dest)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(dest.rglob('*')) if p.is_file()}
 save(dest/'payload_manifest.json',manifest)
 target=ROOT/'HGEN_192_full_reference_8fits_16graphs.zip'
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(dest.rglob('*')):
   if p.is_file():z.write(p,str(p.relative_to(dest)))
 receipt=dict(epoch=time.time(),seconds=time.time()-started,zip=str(target),sha256=sha(target),bytes=target.stat().st_size,files=len(manifest)+1,source_commit='bfbd7b4b864b354fb2dbcb82d1056e3297fc7904',large_weights_uploaded=False)
 save(ROOT/'package_receipt.json',receipt);(ROOT/'ZIP_SHA256.txt').write_text(receipt['sha256']+'  '+target.name+'\n');print(receipt)
if __name__=='__main__':main()
