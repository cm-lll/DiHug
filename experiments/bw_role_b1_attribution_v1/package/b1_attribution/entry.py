"""Stdlib-first admission, no output before path/source-pin checks."""
import json
from pathlib import Path
from multiscale_role.entry_guard import validate_output_path,overlap
from multiscale_role.runtime_guard import checked_sha

SHORT='/root/private_data/gnn_projects/role_mt_v1_2_short_20261018T091510Z'
EXPORT='/root/private_data/gnn_projects/role_mt_v1_2_export_20261018T091055Z'


def required_files(short,export):
    short,export=Path(short),Path(export)
    return [short/'summary.json',short/'case_05.json',short/'frozen_objective.json',short/'policy.json',
            export/'six_states.json',export/'original_evaluation.json']


def inspect_inputs(short,export,check=lambda:None):
    rows=[]
    for path in required_files(short,export):
        check()
        try:
            exists=path.is_file();rows.append(dict(path=str(path),exists=exists,sha256=checked_sha(path,check) if exists else None))
        except OSError as exc:rows.append(dict(path=str(path),exists=False,sha256=None,read_error=type(exc).__name__+': '+str(exc)))
    return dict(status='files_present_not_authenticated' if all(r['exists'] for r in rows) else 'missing_real_inputs',
                files=rows,model_calls=0,source_warning='fresh hashes are fingerprints only; compare summary/case against sealed copies before diagnose')


def admit(out,root,short,export,summary_sha,case_sha,check=lambda:None):
    check()
    if overlap(out,short):raise ValueError('output overlaps protected original short run')
    validate_output_path('short',out,root,manifest=Path(export)/'six_states.json',
                         evaluation=Path(export)/'original_evaluation.json',check=check)
    missing=[str(p) for p in required_files(short,export) if not p.is_file()]
    if missing:raise ValueError('missing real inputs: '+', '.join(missing))
    for sha in (summary_sha,case_sha):
        if not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('explicit sealed summary and case SHA256 pins required; do not substitute freshly computed hashes')
    for path,expected in ((Path(short)/'summary.json',summary_sha),(Path(short)/'case_05.json',case_sha)):
        if checked_sha(path,check)!=expected:raise ValueError('sealed old summary/case hash mismatch')


def validate_done(out,check=lambda:None):
    out=Path(out);summary=json.loads((out/'summary.json').read_text());check()
    for key in ('original_step_reproduced','gradient_sum_verified','original_metrics_reproduced','input_hash_verification_complete'):
        if summary.get(key) is not True:raise ValueError('B1 completion certificate missing '+key)
    if summary.get('status')!='complete' or summary.get('new_candidate_searches')!=0:raise ValueError('B1-only scope/result differs')
    expected={'attribution.json','genre_pmf_cdf.npz','B1_gradient_evidence.npz'}
    if set(summary.get('output_hashes',{}))!=expected:raise ValueError('missing B1 diagnostic evidence')
    for name,h in summary['output_hashes'].items():
        if checked_sha(out/name,check)!=h:raise ValueError('B1 evidence missing or changed')
