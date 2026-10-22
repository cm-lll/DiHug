"""Freeze only the new handoff copies; original paths are read-only audited."""
import hashlib,json,re,sys,zipfile,tempfile,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent;PAYLOAD=ROOT/'payload'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    start=time.perf_counter();archive=ROOT/'DiHug_table2_assets_20261022_v1.zip'
    if archive.exists():raise FileExistsError('immutable release, no overwrite')
    files=[p for p in sorted(PAYLOAD.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='SHA256SUMS']
    for p in files:
        rel=p.relative_to(PAYLOAD)
        if p.suffix in ('.pt','.pth','.ckpt') and str(rel) not in ('reference/imdb/nodes.pt','reference/imdb/edges.pt','reference/pubmed/nodes.pt','reference/pubmed/edges.pt'):raise ValueError('unexpected weight-like file '+str(rel))
        if p.suffix in ('.pem','.key') or p.name.lower() in ('.env','id_rsa','id_ed25519','credentials'):raise ValueError('credential filename rejected')
        if p.suffix in ('.py','.sh','.json','.log','.md','.txt'):
            data=p.read_text(errors='replace')
            if re.search(r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|\bgh[pousr]_[A-Za-z0-9]{30,}|\bAKIA[A-Z0-9]{16}\b',data):raise ValueError('possible credential, stop before packing '+str(rel))
    inventory=json.loads((PAYLOAD/'source_inventory.json').read_text())
    for row in inventory:
        if sha(PAYLOAD/row['path'])!=row['sha256']:raise ValueError('payload does not match copied source '+row['path'])
        if not row['original_path'].startswith('git:') and sha(row['original_path'])!=row['sha256']:raise ValueError('original source changed '+row['original_path'])
    checks=''.join(sha(p)+'  '+str(p.relative_to(PAYLOAD))+'\n' for p in files)
    (PAYLOAD/'SHA256SUMS').write_text(checks)
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[PAYLOAD/'SHA256SUMS']:z.write(p,'DiHug_table2_assets_20261022_v1/'+str(p.relative_to(PAYLOAD)))
    (ROOT/'DiHug_table2_assets_20261022_v1.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
    # Independently extract and verify every payload byte; no expensive second
    # graph evaluation and no model/weight loading.
    dest=Path(tempfile.mkdtemp(prefix='table2-asset-review-',dir=ROOT));folder=dest/'DiHug_table2_assets_20261022_v1'
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('archive CRC error')
        for n in z.namelist():
            if Path(n).is_absolute() or '..' in Path(n).parts:raise ValueError('unsafe ZIP path')
        z.extractall(dest)
    count=0
    for line in (folder/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1)
        if sha(folder/name)!=expected:raise ValueError('independent extraction hash mismatch')
        count+=1
    result=subprocess.run([sys.executable,str(folder/'reproduce.py'),'--help'],capture_output=True,timeout=10)
    if result.returncode:raise ValueError('extracted entry --help failed')
    index=json.loads((folder/'asset_index.json').read_text())
    if index['graph_count']!=80 or index['candidate_graph_count']!=96:raise ValueError('graph inventory mismatch')
    receipt={'status':'passed','archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'checksummed_files':count,'archive_files':count+1,'independent_extraction':str(folder),
        'entry_help_passed':True,'submitted_table2_graphs':80,'additional_candidate_graphs':96,'model_weights_included':False,'reference_pt_files_are_data':True,
        'original_source_hashes_rechecked':True,'CPU_reproduction_graphs':80,'second_full_reproduction_run':False,'wall_seconds':time.perf_counter()-start}
    (ROOT/'release_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
