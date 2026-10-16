"""Independent joint-fusion deployment, bound to CN v2 and no old weights."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parent


def digest(data):return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    base=a.base.resolve();manifest=json.loads((base/'manifest_cn_v2.json').read_text())
    for name,entry in manifest['files'].items():
        if digest((base/name).read_bytes())!=entry['sha256']:raise ValueError('original CN v2 changed: '+name)
    files={}
    for folder in ['hsgd','tests','data','role_degree']:
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ['.py','.json','.npz','']:
                files[str(path.relative_to(ROOT))]=path.read_bytes()
    for name in ['run.py','run.sh','build_release.py','说明.txt','DESIGN_AND_STAGED_PLAN.md','pyg_inspector_compat.diff']:
        if (ROOT/name).exists():files[name]=(ROOT/name).read_bytes()
    frozen=['hsgd/kernel.py','hsgd/noising.py','hsgd/slots.py','hsgd/vendor/transconv_layer.py','hsgd/vendor/layers.py',
        'hsgd/vendor/sample_edges_utils_typed.py','hsgd/bw_preflight/config.json','data/IMDB_movie_reference.npz',
        'data/IMDB_preprocessing_provenance.json','tests/test_contract.py']
    for name in frozen:
        if files[name]!=(base/name).read_bytes():raise ValueError('protected method/data changed: '+name)
    patch=[];changed=[]
    for name,data in sorted(files.items()):
        if name.startswith('hsgd/') and name.endswith('.py'):
            previous=(base/name).read_bytes() if (base/name).is_file() else b''
            if data!=previous:
                changed.append(name)
                patch.extend(difflib.unified_diff(previous.decode().splitlines(True),data.decode().splitlines(True),
                    fromfile='a/'+name if previous else '/dev/null',tofile='b/'+name))
    files['cn_v2_to_joint_v1.diff']=''.join(patch).encode();(ROOT/'cn_v2_to_joint_v1.diff').write_bytes(files['cn_v2_to_joint_v1.diff'])
    for name in ['base_receipt.json','local_acceptance.json']:
        path=ROOT/'results'/name
        if path.exists():
            files['evidence/'+name]=path.read_bytes()
            (ROOT/'evidence').mkdir(exist_ok=True)
            (ROOT/'evidence'/name).write_bytes(files['evidence/'+name])
    for source,name in [('final_tests04/tests.xml','tests.xml'),('final_tests04.log','tests.log'),
                        ('toy03/mechanism.json','mechanism.json'),('toy03/launcher.json','toy_launcher.json'),
                        ('patch_dry_run.json','patch_dry_run.json')]:
        path=ROOT/'results'/source
        if path.is_file():
            files['evidence/'+name]=path.read_bytes()
            (ROOT/'evidence').mkdir(exist_ok=True)
            (ROOT/'evidence'/name).write_bytes(path.read_bytes())
    role_base=ROOT.parent/'role_degree_design_v1'
    role_preserved={}
    for name in ['role_guidance.py','role_statistics.py','role_sampling.py','implementation_contract.json']:
        key='role_degree/'+name
        if files[key]!=(role_base/name).read_bytes():raise ValueError('role degree contract changed: '+name)
        role_preserved[key]=digest(files[key])
    doc=dict(version='bw_joint_cn_v1',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        base_CN_v2_manifest_sha256=digest((base/'manifest_cn_v2.json').read_bytes()),
        protected_method_data_files=frozen,changes=changed,role_degree_unchanged=role_preserved,
        no_old_weight_import=True,no_formal_training_entry=True,no_full_scientific_chain_entry=True,
        files={n:dict(sha256=digest(b),bytes=len(b)) for n,b in files.items()})
    (ROOT/'manifest_joint_cn_v1.json').write_text(json.dumps(doc,indent=2)+'\n')
    files['manifest_joint_cn_v1.json']=(ROOT/'manifest_joint_cn_v1.json').read_bytes()
    checks=''.join(digest(b)+'  '+n+'\n' for n,b in sorted(files.items()));(ROOT/'SHA256SUMS').write_text(checks)
    if a.prepare:print('prepared manifest/patch; no model execution');return
    if a.output is None or 'evidence/local_acceptance.json' not in files:p.error('accepted release and --output required')
    accepted=json.loads(files['evidence/local_acceptance.json'])
    if accepted.get('status')!='passed_engineering_only' or not accepted.get('tests_passed'):raise ValueError('acceptance missing')
    files['SHA256SUMS']=checks.encode();a.output.mkdir(parents=True,exist_ok=False)
    target=a.output/'dihug_sparse_bw_joint_cn_v1.zip'
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for n,b in sorted(files.items()):z.writestr('dihug_sparse_bw_joint_cn_v1/'+n,b)
    sha=digest(target.read_bytes());(a.output/'SHA256SUMS').write_text(sha+'  '+target.name+'\n')
    print(json.dumps(dict(zip=str(target),sha256=sha,bytes=target.stat().st_size,source_commit=doc['source_commit']),indent=2))


if __name__=='__main__':main()
