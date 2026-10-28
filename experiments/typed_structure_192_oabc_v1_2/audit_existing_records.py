"""Read-only CPU receipt audit. No torch import, tensor loading, fit or forward."""
import argparse,hashlib,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'package'))
from admission import dictionaries
from restore_checks import restoration_passes
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def audit(evidence):
    root=pathlib.Path(evidence);expected=json.loads((root/'baseline_expected.json').read_text())
    verified={}
    for rel,digest in expected['evidence_files'].items():
        actual=sha(root/rel)
        if actual!=digest:raise ValueError('fixed baseline evidence differs: '+rel)
        verified[rel]=actual
    binding=json.loads((root/'dictionary_bindings_PROPOSAL_NOT_APPROVAL.json').read_text());dictionaries(root,binding)
    unchanged={}
    for rel,digest in expected['unchanged_source'].items():
        actual=sha(ROOT/rel)
        if actual!=digest:raise ValueError('scientific source differs: '+rel)
        unchanged[rel]=actual
    rows={}
    for d in ['IMDB','PubMed']:
        for a in 'OABC':
            r=json.loads((root/('engineering_'+d)/a/'restore.json').read_text())
            row=json.loads((root/('engineering_'+d)/'summary.json').read_text())['cells'][d+'_'+a]
            passed=r['optimizer_exact'] is True and restoration_passes(r) and r['passes'] is True and row['load_exact'] is True
            rows[d+'_'+a]=dict(passed=passed,optimizer_exact=r['optimizer_exact'],rng_exact=r['rng_exact'],historical_load_exact=row['load_exact'],original_receipt_passed=r['passes'])
            if not passed:raise ValueError('existing restore gate failed: '+d+'_'+a)
    return dict(status='passed_existing_records_only',baseline=expected['baseline'],evidence_sha256=verified,unchanged_source_sha256=unchanged,restore=rows,dictionaries=binding['dictionaries'],pubmed_all_types_hit_Kmax=True,new_optimizer_steps=0,new_gpu_tests=0,new_dictionary_fits=0,formal_updates=0,new_generated_graphs=0)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',default=str(ROOT/'evidence'));a=p.parse_args();print(json.dumps(audit(a.evidence),indent=2))
