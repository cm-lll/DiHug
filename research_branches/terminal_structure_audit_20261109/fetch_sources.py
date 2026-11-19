"""Download a fixed small primary-source set for text inspection only; never import it."""
import hashlib, json, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/terminal_structure_audit_20261109/sources'
SOURCES = {
    'GDPO': ('sail-sg/GDPO', 'ca29d9620f2e4d8cd5ab290bfd70119fc5ee9f77', ['main_ppo.py', 'model/diffusion_discrete.py', 'diffusion/diffusion_utils.py', 'run_ppo_sbm.sh']),
    'GenStat': ('kiarashza/GenStat', '0fda6fde4899f0bfbdff192991c7e7452c22a6e3', ['GlobalPrespective.py', 'GraphGeneration.py', 'README.md']),
}
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows=[]
    for name,(repo,commit,files) in SOURCES.items():
        for f in files:
            dst=OUT/name/f
            if dst.exists():
                raise FileExistsError(dst)
            url=f'https://raw.githubusercontent.com/{repo}/{commit}/{f}'
            row=dict(repository=repo,commit=commit,file=f,url=url)
            try:
                req=urllib.request.Request(url,headers={'User-Agent':'DiHug-read-only-research'})
                data=urllib.request.urlopen(req,timeout=20).read(2000001)
                if len(data)>2000000: raise ValueError('file size bound')
                dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data)
                row.update(status='downloaded_not_executed',bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            except Exception as e:
                row.update(status='unavailable',error=repr(e))
            rows.append(row)
    (OUT/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows))
if __name__=='__main__':main()
