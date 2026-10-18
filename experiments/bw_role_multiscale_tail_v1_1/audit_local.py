"""Local read-only source audit and SMALL ENGINEERING numeric witnesses."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'package'),str(ROOT/'package/tests')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--legacy-zip',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    start=time.monotonic()
    import numpy as np
    import scipy
    import torch
    import torch_geometric
    import pytest
    from multiscale_role.legacy import source_archive,convert_arrays,extract_frequency,baseline_alignment,ZIP_SHA
    from multiscale_role.evaluation import genre64,degree_w1
    from multiscale_role.core import MultiScaleGuide,freeze
    from role_degree.role_statistics import joint_profile
    from test_legacy_repairs import fixture
    import legacy_genre_reference as old
    files,pins,inputs=source_archive(a.legacy_zip)
    retained=(ROOT/'package/tests/legacy_genre_reference.py').read_bytes()
    identical_AST=ast.dump(ast.parse(retained),include_attributes=False)==ast.dump(ast.parse(files['genre_diagnostics.py']),include_attributes=False)
    if not identical_AST:raise ValueError('retained independent evaluator changed')
    case,l,b,col,cp=fixture();values,origins,err,qhash=convert_arrays(case,b,col,cp,20)
    pf=joint_profile(l.types,case['clean'],case['space'].encode(case['clean'])[0],l.pairs)
    guide=MultiScaleGuide(l,freeze(pf,l,b['alpha']));group=pf['groups']['3']
    degrees=np.repeat(np.asarray(group['rows'])[:,8],group['counts'])
    report=old.diagnose_genre_arrays(b['q0'],l.types,l.slots,l.families,l.pairs,l.prior,guide.alpha[19],degrees,np.zeros((32,10)))
    w,path=extract_frequency(report,l)
    metrics=dict(Genre64=genre64(b['q0'],guide,19,dict(Genre_type=3,omega64=w,CF_normalizer=1)),
                 expected_degree_PMF_W1=degree_w1(b['q0'],guide,19,only_types={'3'}))
    result=dict(status='passed_scoped_source_and_synthetic_audit',source_zip_sha256=ZIP_SHA,
        source_release_verified_files=len(json.loads(files['release.json'])),
        retained_original_evaluator_AST_identical=identical_AST,
        audited_source_files={k:hashlib.sha256(files[k]).hexdigest() for k in ('worker.py','genre_diagnostics.py','original_pins.json','input_pins.json')},
        fixture=dict(nodes=len(l.types),slots=len(l.families),actual_IMDB_frozen_state=False),
        synthetic_original_q0_bridge_max_error=err,synthetic_query_digest_equal=qhash==cp['query_sha256'],
        independent_old_Genre_replay=baseline_alignment(metrics,report),field_origins=origins,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,pyg=torch_geometric.__version__,pytest=pytest.__version__),
        real_six_states_executed=False,BW_connected=False,scientific_model_calls=0,training_updates=0,scientific_graphs=0,GPU_use=0,
        actual_old_arrays_reports_and_collections_locally_available=False,wall_seconds=time.monotonic()-start)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
