"""Release preparation only; no runtime dependency installation or model calls."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent;P=R/'package'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    files={str(p.relative_to(P)):sha(p) for p in sorted(P.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json' and '.pytest_cache' not in p.parts}
    (P/'manifest.json').write_text(json.dumps({'version':'imdb_role_pair_gpu_v1_1','repair_baseline':'bb32bfaa5b63f049cbcba4408e2184a39e5c8b6a','model_baseline':'f4d410cc37a76b065634eadb27a5421d53f638a0','files':files},indent=2)+'\n')
    print('frozen',len(files),'files; manifest',sha(P/'manifest.json'))
if __name__=='__main__':main()
