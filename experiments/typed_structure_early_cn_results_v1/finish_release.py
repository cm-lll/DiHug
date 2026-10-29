"""Curves, sealed ZIP, independent extraction and CPU public re-audit."""
import hashlib,json,os,pathlib,shutil,subprocess,sys,time,zipfile
P=pathlib.Path(__file__).resolve().parent;OUT=P/'release_payload'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    started=time.time();curves=OUT/'curves';curves.mkdir(exist_ok=False)
    for d,seed in [('IMDB',2026102911),('PubMed',2026102912)]:
        folder=OUT/'run'/f'{d}_s{seed}_E'
        rows=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()]
        vals=json.loads((folder/'validation.json').read_text())
        fig,ax=plt.subplots(1,2,figsize=(10,3.5))
        losses=np.array([x['loss'] for x in rows]);ax[0].plot(range(1,len(rows)+1),losses,alpha=.2,linewidth=.5,label='normal2 update loss')
        ax[0].plot(range(50,len(rows)+1),np.convolve(losses,np.ones(50)/50,'valid'),label='50-update mean')
        for kind in ['raw','ema_0.99']:
            for t in [1,500,1000]:
                ax[1].plot([v['updates'] for v in vals],[next(r['ce_sum_over_S'] for r in v[kind]['rows'] if r['t']==t) for v in vals],marker='o',label=f'{kind} t={t}')
        for a in ax:a.set_xlabel('normal2 updates');a.set_ylabel('CE / full schema S');a.legend(fontsize=6)
        fig.suptitle(f'{d} E-earlyCN train seed {seed}');fig.tight_layout();fig.savefig(curves/f'{d}_training_validation.png',dpi=160);plt.close(fig)
    shutil.copy2(__file__,OUT/'finish_release.py')
    index={str(f.relative_to(OUT)):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name not in ['PAYLOAD_SHA256.json','SHA256SUMS']}
    (OUT/'PAYLOAD_SHA256.json').write_text(json.dumps(index,indent=2));(OUT/'SHA256SUMS').write_text(''.join(v['sha256']+'  '+k+'\n' for k,v in index.items()))
    archive=P/'typed_structure_early_cn_results_v1.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob('*')):
            if f.is_file():z.write(f,str(f.relative_to(OUT)))
    scratch=P/'release_unpack_check';scratch.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        for m in z.infolist():
            assert scratch.resolve() in (scratch/m.filename).resolve().parents
            assert (m.external_attr>>16)&0o170000!=0o120000
        z.extractall(scratch)
    for name,value in index.items():
        f=scratch/name;assert f.stat().st_size==value['bytes'] and sha(f)==value['sha256'],name
    src=scratch/'source_unpacked';src.mkdir()
    with zipfile.ZipFile(scratch/'runtime_source.zip') as z:
        for m in z.infolist():assert src.resolve() in (src/m.filename).resolve().parents
        z.extractall(src)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    subprocess.run([sys.executable,str(scratch/'audit_completed.py'),'--run',str(scratch/'run'),'--source',str(src/'typed_structure_early_cn_v1'),'--audit-library',str(scratch/'audit_library.py'),'--output',str(scratch/'public_audit.json'),'--public-only'],check=True,env=env,timeout=180)
    receipt=dict(status='passed',files=len(index),zip_sha256=sha(archive),zip_bytes=archive.stat().st_size,zip_path=str(archive),independent_extraction=True,public_audit=json.loads((scratch/'public_audit.json').read_text()),seconds=time.time()-started,finished_epoch=time.time())
    (P/'archive_verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:receipt[k] for k in ['status','files','zip_sha256','zip_bytes','seconds']}))
if __name__=='__main__':main()
