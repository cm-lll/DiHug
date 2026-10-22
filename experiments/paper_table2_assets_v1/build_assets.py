"""Read-only historical asset discovery/copy. Never runs models or generators."""
import hashlib,json,subprocess,sys,time,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OLD=Path('/data2/lyh/gnn_project/DiHug_clean_joint')
HGEN=Path('/data2/lyh/gnn_project/HGEN')
PROJECT=Path('/data2/lyh/gnn_project/DiHug')
OUT=ROOT/'payload'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def main():
    start=time.perf_counter()
    if OUT.exists():raise FileExistsError('independent new output required: '+str(OUT))
    OUT.mkdir();inventory=[];weights=[];groups=[]
    def copy(p,rel,kind):
        p=Path(p).resolve();q=OUT/rel
        before=sha(p);q.parent.mkdir(parents=True,exist_ok=True)
        if q.exists():
            if sha(q)!=before:raise ValueError('destination conflict '+str(q))
        else:shutil.copyfile(p,q)
        if sha(p)!=before or sha(q)!=before:raise ValueError('source changed or copy mismatch '+str(p))
        row={'original_path':str(p),'path':str(q.relative_to(OUT)),'sha256':before,'bytes':p.stat().st_size,'kind':kind}
        inventory.append(row);return row
    def oldcopy(p,kind='original_evidence'):
        p=Path(p);return copy(p,'original_reports/'+str(p.relative_to(OLD)),kind)
    def weight(p,context):
        p=Path(p);row={'path':str(p),'context':context,'uploaded':False,'exists':p.is_file()}
        if p.is_file():row.update(bytes=p.stat().st_size,sha256=sha(p))
        weights.append(row)
    provenance=OLD/'paper/results_20260727/provenance.json'
    for rel in ('paper/results_20260727/provenance.json','paper/results_20260727/fullgraph_core_8run.tsv','paper/results_20260727/fullgraph_distribution_8run.tsv','paper/results_20260729/README.md','paper/results_20260729/fullgraph_dihug_extended.tsv','paper/results_20260729/diffusion_ablation.tsv','paper/aaai2027_dihug.tex','environment.yml','requirements.txt','FINAL_PROTOCOL.md'):
        oldcopy(OLD/rel)
    copy(PROJECT/'temp/35541_DiHug_Typed_Edit_Diffusi (2).pdf','paper/submitted.pdf','submitted_pdf')
    # Entire original evaluation module tree, unchanged. These definitions are
    # imported for CPU statistics only; no generation entry point is executed.
    for p in sorted((OLD/'sparse_diffusion/structure_energy').glob('*.py')):
        copy(p,'original_source/'+str(p.relative_to(OLD)),'original_python_source')
    copy(OLD/'sparse_diffusion/__init__.py','original_source/sparse_diffusion/__init__.py','original_python_source')
    scripts=('evaluate_heterogeneous_generation.py','evaluate_external_graphs_canonical.py','generate_extended_structure_tables.py','build_consistent_paper_evidence.py','run_classic_heterogeneous_baseline.py','run_current_base_diffusion_ablation_8gpu.sh','run_hgen_fullgraph_multiseed_8gpu.sh','run_seeded_hgen.py','parse_hgen_baseline_log.py','summarize_hgen_multiseed.py','run_classic_baseline_real_followup_4gpu.sh','run_classic_baseline_extension_8gpu.sh','run_strict_edit_diffusion_multiseed_gate_8gpu.sh')
    for name in scripts:
        p=OLD/'scripts'/name
        if p.exists():copy(p,'original_source/scripts/'+name,'original_script_DO_NOT_train_or_generate')
    for name in ('convert_hgen_graph_to_subgraphs.py','convert_hgen_pubmed_baseline.py'):
        p=OLD/'dataprocess/scripts'/name;copy(p,'original_source/dataprocess/scripts/'+name,'actual_preprocessing_source')
    # Keep raw benchmark serialization and original HGEN node-order evidence.
    for dataset,folder,hfolder in (('imdb','IMDB_movie_subgraphs','IMDB_movie'),('pubmed','PubMed_baseline_subgraphs','PubMed')):
        for name in ('nodes.pt','edges.pt','meta.json'):
            copy(OLD/'data'/folder/'subgraph_000'/name,'reference/'+dataset+'/'+name,'original_reference')
        for name in ('new_node_dict.p','adj_matrix.p'):
            copy(HGEN/hfolder/name,'hgen_reference/'+dataset+'/'+name,'original_HGEN_reference_and_node_order')
        roots={
            'DiHug':OLD/'results/diffusion_ablation_current_base_0729/extended_structure'/f'{dataset}_main.json',
            'GraphRNN':OLD/'results/classic_baseline_real_followup'/f'{dataset}_graphrnn/report.json',
            'NetGAN':OLD/'results/classic_baseline_extension'/f'{dataset}_netgan/report.json',
            'VGAE':OLD/'results/classic_baseline_extension'/f'{dataset}_vgae/report.json',
            'HGEN':OLD/'results/hgen_fullgraph_eo_audit_0729'/f'{dataset}.json'}
        for method,report in roots.items():
            x=json.loads(report.read_text());copied=oldcopy(report,'original_per_graph_and_aggregate_report');records=[]
            if len(x['rows'])!=8:raise ValueError('expected all eight paper samples')
            for i,row in enumerate(x['rows']):
                raw=Path(row['source'] if method=='HGEN' else row['path'])
                source=raw if raw.is_absolute() else OLD/raw
                sid=int(row.get('seed',row.get('sample',i)))
                graph=copy(source,f'graphs/{dataset}/{method}/sample{sid:03d}/'+source.name,'original_generated_graph')
                extra=[]
                if method=='DiHug':
                    for name in ('sampling_state.json','gate_report.json','exchangeable_role_degree_prior.json'):
                        extra.append(copy(source.with_name(name),f'graphs/{dataset}/{method}/sample{sid:03d}/'+name,'original_sampling_metadata')['path'])
                elif method=='HGEN':
                    for name in ('run.log','report.json'):
                        if source.with_name(name).is_file():extra.append(copy(source.with_name(name),f'graphs/{dataset}/{method}/sample{sid:03d}/'+name,'original_HGEN_seed_receipt')['path'])
                    if json.loads(source.read_text())['run_seed']!=sid:raise ValueError('HGEN seed mismatch')
                records.append({'sample':sid,'original_path':str(source),'path':graph['path'],'sha256':graph['sha256'],'original_metadata':extra,
                    'seed_semantics':'generation seed' if method=='DiHug' else 'independent HGEN training/run seed' if method=='HGEN' else 'sample index; training seed0; selection RNG 0*100003+index according to retained source'})
            if sorted(r['sample'] for r in records)!=list(range(8)):raise ValueError('incomplete/duplicate sample inventory')
            groups.append({'dataset':dataset,'method':method,'version':'submitted Table2, provenance + exact row tables; pending current CPU reproduction','original_report':copied['path'],'graphs':records})
            if method not in ('DiHug','HGEN'):weight(report.parent/'model.pt',dataset+'/'+method)
        # Original DiHug training record and the historical script's base model.
        modeldir=OLD/'results/strict_edit_diffusion_gate_0721'/(dataset+'_strict_cal5')
        weight(modeldir/'mixed_edit_diffusion.pt',dataset+'/DiHug; referenced by retained launcher; per-output checkpoint hash not recorded')
        oldcopy(modeldir/'train_report.json','original_training_report')
        # This independent retained artifact supplies an additional actual hash.
        if dataset=='pubmed':oldcopy(OLD/'results/pubmed_base_main8_0728/aggregate.json')
        for p in sorted((OLD/'logs/diffusion_ablation_current_base_0729').glob(dataset+'_seed*_main.log')):oldcopy(p,'original_generation_log')
        for method,dirname in (('GraphRNN','classic_baseline_real_followup'),('NetGAN','classic_baseline_extension'),('VGAE','classic_baseline_extension')):
            short={'GraphRNN':'graphrnn','NetGAN':'netgan','VGAE':'vgae'}[method];p=OLD/'logs'/dirname/f'{dataset}_{short}.log'
            if p.exists():oldcopy(p,'original_training_log')
    for rel in ('results/diffusion_ablation_current_base_0729/aggregate.json','results/diffusion_ablation_current_base_0729/paper_tables/sample_ablation.tsv','results/hgen_fullgraph_multiseed_audit_0727/aggregate.json','results/hgen_fullgraph_canonical_8run_0727/aggregate.json'):
        oldcopy(OLD/rel)
    # Register conflicting snapshots explicitly; they are not used as Table2.
    histories=[]
    for root in ('paper_final_context_binomial_xiaob','paper_final_context_binomial_xiaob_s8_15','paper_final_context_xiaob'):
        p=PROJECT/'results'/root/'aggregate.json'
        if p.exists():histories.append(copy(p,'candidate_history/'+root+'/aggregate.json','candidate historical version; NOT submitted Table2'))
    history_source='81ac099d5bef7da08dedc2c22d2626b113eede9e5dd08abb3acac70add9e23b5'
    hgen_source={'path':str(HGEN/'hgen.py'),'historically_recorded_sha256':history_source,'current_sha256':sha(HGEN/'hgen.py'),'exact_historical_source_found':False}
    for commit in subprocess.check_output(['git','-C',str(HGEN),'log','--all','--format=%H','--','hgen.py'],text=True).splitlines():
        content=subprocess.check_output(['git','-C',str(HGEN),'show',commit+':hgen.py'])
        if hashlib.sha256(content).hexdigest()==history_source:
            p=OUT/'original_source/HGEN/hgen.py';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content);hgen_source.update(exact_historical_source_found=True,git_commit=commit)
    # Current HGEN source is registered, not misrepresented as the historical one.
    source={'old_repo_commit':subprocess.check_output(['git','-C',str(OLD),'rev-parse','HEAD'],text=True).strip(),'old_repo':'DiHug_clean_joint','current_HGEN_source':hgen_source,
        'generation_runtime_source_binding':'DiHug launchers and reports retained; no per-graph authenticated checkpoint hash receipt for every method; not claimed complete',
        'no_models_executed':True,'no_training':True,'no_new_graphs':True,'historical_results_modified':False}
    write(OUT/'asset_index.json',{'groups':groups,'graph_count':sum(len(g['graphs']) for g in groups),'datasets':['imdb','pubmed'],'scope':'all eight submitted Table2 samples per method; other historical experiments not claimed exhaustive','candidate_histories':histories})
    write(OUT/'source_binding.json',source);write(OUT/'weight_registry.json',weights);write(OUT/'source_inventory.json',inventory)
    print(json.dumps({'copied_files':len(inventory),'graphs':80,'seconds':time.perf_counter()-start,'hgen_source':hgen_source},ensure_ascii=False))

if __name__=='__main__':main()
