"""Four forward-only calls per dataset to explain a failed bitwise parity gate.

Diagnostic evidence only; does not change the acceptance gate or update models.
"""
from . import gpu_engineering as g


def main():
    ap=g.argparse.ArgumentParser();ap.add_argument('--assets-root',type=g.Path,required=True)
    ap.add_argument('--output',type=g.Path,required=True);args=ap.parse_args()
    args.output.mkdir(exist_ok=False)
    g.torch.set_num_threads(1);g.torch.set_num_interop_threads(1)
    g.torch.backends.cuda.matmul.allow_tf32=False;g.torch.backends.cudnn.allow_tf32=False
    tasks=g.json.loads((args.assets_root/'results/cn_multiseed_20261104T042546/tasks.json').read_text())
    records=[];begin=g.time.time()
    for dataset,seed in [('IMDB',2026110411),('PubMed',2026110421)]:
        row=next(x for x in tasks if x['dataset']==dataset and x['train_seed']==seed and x['arm']=='A');cfg=row['config']
        entry=cfg['datasets'][dataset];path=args.assets_root/g.SCI.relative_to(g.ROOT)/'package'/entry['file']
        assert g.hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        with g.np.load(path,allow_pickle=False) as z:case=g.common.make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
        path=args.assets_root/'results/cn_multiseed_20261104T042546/run'/f'{dataset}_{seed}_A_late/initial.pt'
        registry=g.json.loads((g.ROOT/'review/structure_input_v1/ASSET_BINDINGS.json').read_text())['assets']
        asset=next(x for x in registry if x['kind']=='checkpoint_bytes_only' and x['dataset']==dataset and x['seed']==seed and x['update']==0)
        assert g.hashlib.sha256(path.read_bytes()).hexdigest()==asset['actual_sha256']
        p=g.torch.load(path,map_location='cpu');assert p['updates']==0
        a=g.StructuralDenoiser(case,cfg,'A');a.load_state_dict(p['model'],strict=True)
        b=g.StructureInputA.from_initial_A(a,case,cfg,completed_updates=0)
        assert all(g.torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
        a.cuda().eval();b.cuda().eval()
        rng=g.np.random.default_rng(1);g.rng_restore(p['rng'],rng,'cuda')
        inp=g.one_input(case,cfg,1,rng,force_t=25);saved=g.rng_capture(rng,'cuda')
        arrays={};modes={}
        for name,model in [('A1',a),('A2',a),('candidate1',b),('candidate2',b)]:
            g.rng_restore(saved,rng,'cuda')
            with g.torch.no_grad():
                ctx=g.FrozenTypedContext(model,case['space'].types,inp['edges'],inp['edge_families'],name)
                try:_,logits,_=ctx.predict(inp['query'],inp['f'],.025,name,details=True)
                finally:ctx.close()
            arrays[name]=logits.cpu();modes[name]=g.digest(g.rng_capture(rng,'cuda'))==g.digest(saved)
        def compare(x,y):
            left,right=arrays[x],arrays[y];diff=(left-right).abs()
            return dict(left=x,right=y,bitwise_equal=g.torch.equal(left,right),max_abs=float(diff.max()),
                        nonzero_elements=int((diff>0).sum()),total_elements=diff.numel(),
                        allclose_original_config=g.torch.allclose(left,right,atol=cfg['tolerances']['fp32_atol'],rtol=cfg['tolerances']['fp32_rtol']))
        records.append(dict(dataset=dataset,RNG_unchanged=modes,tolerances=cfg['tolerances'],
                            comparisons=[compare(x,y) for x,y in [('A1','A2'),('candidate1','candidate2'),('A1','candidate1')]]))
        g.torch.save(dict(input=inp,logits=arrays),args.output/(dataset+'_parity_arrays.pt'))
        del a,b,p;g.gc.collect();g.torch.cuda.empty_cache()
    g.save(args.output/'result.json',dict(status='diagnostic_only',start_epoch=begin,end_epoch=g.time.time(),records=records))


if __name__=='__main__':main()
