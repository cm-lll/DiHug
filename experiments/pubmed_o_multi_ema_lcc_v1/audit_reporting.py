"""Bounded independent checks of the two-sample and old2/new6/all8 summaries."""
import collections,csv,json,math,pathlib,statistics,time
ROOT=pathlib.Path(__file__).parent
def read(name):return list(csv.DictReader((ROOT/name).open()))
def val(x):return None if x=='' else float(x)
def equal(a,b):
    assert (a is None)==(b is None),(a,b)
    if a is not None:assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-11),(a,b)
def main():
    start=time.time();ledger=json.loads((ROOT/'ledger.json').read_text());assert start<ledger['deadline_epoch']-60;checks=0
    base=collections.defaultdict(list)
    for row in read('lcc_minus_original.csv'):base[row['train_seed'],row['readout'],row['metric_group'],row['metric']].append(val(row['difference']))
    for row in read('cleaning_seed_means.csv'):
        values=base[row['train_seed'],row['readout'],row['metric_group'],row['metric']];assert len(values)==2
        expected=None if any(v is None for v in values) else sum(values)/2;equal(val(row['value']),expected);checks+=1
    seed_rows=collections.defaultdict(list)
    for row in read('cleaning_seed_means.csv'):seed_rows[row['readout'],row['metric_group'],row['metric']].append(row)
    for row in read('cleaning_cohorts.csv'):
        values=[val(r['value']) for r in seed_rows[row['readout'],row['metric_group'],row['metric']] if row['cohort']=='all_eight' or r['cohort']==row['cohort']];assert len(values)==int(row['expected_seeds']);expected=None if any(v is None for v in values) else sum(values)/len(values)
        equal(val(row['mean_difference']),expected);equal(val(row['median_difference']),None if expected is None else statistics.median(values));checks+=2
        if expected is None:assert row['negative_count']==row['zero_count']==row['positive_count']==''
        else:assert [int(row[k]) for k in ['negative_count','zero_count','positive_count']]==[sum(v<0 for v in values),sum(v==0 for v in values),sum(v>0 for v in values)]
    rel=read('relations.csv');index={(r['train_seed'],r['readout'],r['sample_seed'],r['version'],r['family']):r for r in rel}
    for row in read('relation_retention_summary.csv'):
        original=[r for r in rel if r['readout']==row['readout'] and r['family']==row['family'] and r['version']=='original'];pairs=[(r,index[r['train_seed'],r['readout'],r['sample_seed'],'lcc',r['family']]) for r in original]
        a,b=sum(int(x['edges']) for x,y in pairs),sum(int(y['edges']) for x,y in pairs);equal(val(row['pooled_edge_retention']),b/a if a else None)
        ratios=[int(y['edges'])/int(x['edges']) if int(x['edges']) else None for x,y in pairs];equal(val(row['mean_pergraph_retention']),sum(ratios)/len(ratios) if all(v is not None for v in ratios) else None);checks+=2
    result=dict(status='passed',checks=checks,seconds=time.time()-start,scope='cleaning perseed means, cohort medians/signs, strict null propagation, pooled-vs-pergraph relation retention; CPU only')
    (ROOT/'reporting_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
