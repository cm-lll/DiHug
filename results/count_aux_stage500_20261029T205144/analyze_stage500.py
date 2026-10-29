"""Read-only CPU reporting of frozen stage500 outputs; never loads a model."""
import pathlib,json,csv,hashlib,statistics,datetime,sys
ROOT=pathlib.Path(__file__).resolve().parent
RUN=ROOT/'run'
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def main():
    panel=read(RUN/'unified_evaluation.json')
    out=ROOT/'analysis';out.mkdir(exist_ok=True)
    rows=panel['validation'];rels=panel['family'];pairs=panel['paired_family'];grads=panel['gradients']
    audit=dict(generation_files=list(map(str,RUN.rglob('graph.npz'))),pairing=panel['pairing'],checkpoint_files=[],validation_input_consistency=[],ce_reconstruction=[])
    for cell in panel['cells']:
        p=RUN/cell['id']
        for u in [0,100,250,500]:
            file=p/('initial.pt' if u==0 else f'checkpoint_u{u}.pt')
            receipt=p/('initial.sha256.json' if u==0 else f'checkpoint_u{u}.sha256.json')
            actual=digest(file) if file.exists() else None
            expected=read(receipt)['sha256'] if receipt.exists() else None
            audit['checkpoint_files'].append(dict(task=cell['id'],updates=u,path=str(file),bytes=file.stat().st_size if file.exists() else None,sha256=actual,receipt_sha256=expected,passed=actual is not None and actual==expected))
    for ds in ['IMDB','PubMed']:
        seeds=sorted({r['train_seed'] for r in rows if r['dataset']==ds})
        for seed in seeds:
            for t in [1,500,1000]:
                hashes={r['input_sha256'] for r in rows if (r['dataset'],r['train_seed'],r['t'])==(ds,seed,t)}
                audit['validation_input_consistency'].append(dict(dataset=ds,seed=seed,t=t,unique_inputs=len(hashes),passed=len(hashes)==1))
    for r in rows:
        f=[x for x in rels if all(x[k]==r[k] for k in ['dataset','train_seed','arm','update','readout','t'])]
        ce=sum((x['positive_ce'] or 0)*x['positive_count']+(x['negative_ce'] or 0)*x['negative_count'] for x in f)/18336
        audit['ce_reconstruction'].append(dict(dataset=r['dataset'],seed=r['train_seed'],arm=r['arm'],update=r['update'],readout=r['readout'],t=r['t'],difference=ce-r['ce']))
    audit['max_ce_reconstruction_abs_difference']=max(abs(r['difference']) for r in audit['ce_reconstruction']) if rows else None
    train_math=[];relation_rows=[];gradient_inputs=[]
    for cell in panel['cells']:
        logfile=RUN/cell['id']/'training.jsonl'
        if not logfile.exists():continue
        ident={k:cell[k] for k in ['dataset','train_seed','arm']}
        names=['GG','GD','GC','GS','DD','DC','DS','CC','CS','SS'] if cell['dataset']=='PubMed' else ['DD','DA','DM','DG','AA','AM','AG','MM','MG','GG']
        for line in logfile.read_text().splitlines():
            r=json.loads(line);shadow_sum=actual_sum=0.
            if r.get('gradient_observation'):
                obs=r['gradient_observation']
                for group,st in {'ALL':obs['all_parameters'],**obs['disjoint_groups']}.items():
                    gradient_inputs.append(dict(**ident,update=r['update'],t0=r['t'][0],t1=r['t'][1],input_pair_sha256=r['input_sha256'],actual_lambda=r['lambda_count'],group=group,**st))
            for i,inp in enumerate(r['inputs']):
                c=inp['counts'];den=c['relation_denominator']
                raw_penalties=[rho if included else 0. for rho,included in zip(c['normalized_penalty'],c['included'])]
                total=sum(raw_penalties)
                for f,rho in enumerate(raw_penalties):
                    shadow=.001*rho/den if den else 0.;actual=r['lambda_count']*rho/den if den else 0.
                    shadow_sum+=shadow/2;actual_sum+=actual/2
                    relation_rows.append(dict(**ident,update=r['update'],input_index=i,t=inp['t'],input_pair_sha256=r['input_sha256'],family=f,relation=names[f],
                        mu=c['expected_edges'][f],target=c['target_edges'][f],delta=c['signed_error'][f],
                        scale=c['residual_scale'][f],standardized_residual=c['standardized_residual'][f],penalty=rho,
                        included=c['included'][f],relation_denominator=den,shadow_input_contribution=shadow,
                        actual_half_input_contribution=actual/2,share=rho/total if total>0 else None))
            train_math.append(dict(task=cell['id'],update=r['update'],loss_error=r['loss']-r['ce']-r['weighted_auxiliary'],
                contribution_error=actual_sum-r['weighted_auxiliary'],shadow_error=shadow_sum-.001*r['auxiliary'],optimizer_steps=r['optimizer_steps'],ema_updates=r['ema_updates']))
    if relation_rows:
        with (out/'training_relation.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(relation_rows[0]));w.writeheader();w.writerows(relation_rows)
    if gradient_inputs:
        with (out/'gradient_observation_with_inputs.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(gradient_inputs[0]));w.writeheader();w.writerows(gradient_inputs)
    audit['training_arithmetic']=dict(rows=len(train_math),maximum_loss_error=max((abs(x['loss_error']) for x in train_math),default=None),
        maximum_relation_contribution_error=max((abs(x['contribution_error']) for x in train_math),default=None),
        maximum_shadow_error=max((abs(x['shadow_error']) for x in train_math),default=None),
        counters_correct=all(x['optimizer_steps']==1 and x['ema_updates']==x['update'] for x in train_math))
    audit['passed']=not audit['generation_files'] and all(x['passed'] for x in audit['checkpoint_files']+audit['validation_input_consistency']) and all(x['initial_exact'] and x['reset_rng_exact'] and x['input_pairing_complete'] for x in panel['pairing']) and audit['training_arithmetic']['counters_correct'] and all(abs(x[k])<1e-9 for x in train_math for k in ['loss_error','contribution_error','shadow_error'])
    dump(out/'independent_audit.json',audit)
    focus=[r for r in pairs if r['update'] in [100,250,500] and ((r['dataset']=='PubMed' and r['relation'] in ['GD','DC','SS']) or (r['dataset']=='IMDB' and r['relation']=='AA'))]
    dump(out/'focus_relation_pairs.json',focus)
    # Actual weighted penalty contributions, separately from module gradients.
    contribution=[]
    for row in rows:
        f=[x for x in rels if all(x[k]==row[k] for k in ['dataset','train_seed','arm','update','readout','t'])]
        total=sum(x['shadow_aux_contribution'] for x in f)
        for x in f:
            contribution.append(dict(**{k:x[k] for k in ['dataset','train_seed','arm','update','readout','t','relation']},weighted_contribution=x['shadow_aux_contribution'],share=x['shadow_aux_contribution']/total if total>0 else None,shadow_only=x['arm']=='CE'))
    dump(out/'relation_penalty_shares.json',contribution)
    # CPU plotting; no extra prediction or optimizer operations.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    identities=sorted({(x['dataset'],x['train_seed']) for x in rows})
    fig,axs=plt.subplots(4,3,figsize=(12,13),squeeze=False)
    for i,(ds,seed) in enumerate(identities):
        for j,t in enumerate([1,500,1000]):
            ax=axs[i,j]
            for arm in ['CE','CE_COUNT']:
                a=sorted([x for x in rows if (x['dataset'],x['train_seed'],x['arm'],x['readout'],x['t'])==(ds,seed,arm,'ema_0.99')],key=lambda x:x['update'])
                ax.plot([x['update'] for x in a],[x['sum_relation_absolute_error'] for x in a],marker='o',label=arm)
            ax.set_title(f'{ds} seed{seed}, t={t}');ax.set_xlabel('update');ax.set_ylabel('sum relation absolute count error');ax.legend()
    fig.tight_layout();fig.savefig(out/'ema_validation_relation_count_error.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for ax,(ds,seed) in zip(axs.flat,identities):
        for arm in ['CE','CE_COUNT']:
            a=sorted([x for x in grads if (x['dataset'],x['train_seed'],x['arm'],x['group'])==(ds,seed,arm,'ALL')],key=lambda x:x['update'])
            ax.plot([x['update'] for x in a],[x['auxiliary_to_ce_l2'] for x in a],marker='o',label=arm)
        ax.set_title(f'{ds} {seed}');ax.set_xlabel('update');ax.set_ylabel('weighted auxiliary / CE parameter gradient L2');ax.legend()
    fig.tight_layout();fig.savefig(out/'parameter_gradient_ratio.png',dpi=140);plt.close(fig)
    print(json.dumps(dict(panel_status=panel['status'],audit_passed=audit['passed'],checkpoints=len(audit['checkpoint_files']),validation_rows=len(rows),relation_rows=len(rels),gradient_rows=len(grads),max_ce_error=audit['max_ce_reconstruction_abs_difference']),indent=2))
if __name__=='__main__':main()
