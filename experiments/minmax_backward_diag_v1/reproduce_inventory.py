"""Read-only inventory, no model construction/forward/optimizer/GPU use."""
import argparse,hashlib,json,pathlib
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--v2-results',required=True);a=ap.parse_args()
    root=pathlib.Path(a.v2_results).resolve()
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    print(json.dumps({'checkpoints':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(root.rglob('*.pt'))],
        'note':'A filename or matching seed is not initial-weight authentication; original v2 source only saved after1.pt.'},indent=2))
if __name__=='__main__':main()
