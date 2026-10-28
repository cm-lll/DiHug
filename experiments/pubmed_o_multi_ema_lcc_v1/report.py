"""Generate fixed all-readout summary tables from completed CPU statistics."""
import collections,csv,json,pathlib,statistics
ROOT=pathlib.Path(__file__).parent
SEEDS=[2026102812,2026102833,2026102841,2026102842,2026102843,2026102844,2026102845,2026102846]
KINDS=['raw','ema_0.9','ema_0.97','ema_0.99']
def read(name):return list(csv.DictReader((ROOT/name).open()))
def number(v):return None if v=='' else float(v)
def avg(values):return statistics.mean(values) if values and all(v is not None for v in values) else None
def fmt(v):return 'null' if v is None else f'{v:.4f}' if isinstance(v,float) else str(v)
def table(header,rows):return '\n'.join(['| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in rows])
def main():
    cells=read('seed_means.csv');ret=read('retention.csv');cohort=read('cohort_pairs.csv');div=read('diversity.csv');excluded=json.loads((ROOT/'excluded_components.json').read_text());summary={};allmetrics={}
    def write_csv(name,rows):
        with (ROOT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    rel=read('relations.csv');ri={(r['train_seed'],r['readout'],r['sample_seed'],r['version'],r['family']):r for r in rel};relations=[]
    for kind in KINDS:
        for family in map(str,range(10)):
            rows=[r for r in rel if r['readout']==kind and r['version']=='original' and r['family']==family];pairs=[(r,ri[r['train_seed'],kind,r['sample_seed'],'lcc',family]) for r in rows];a=sum(int(x['edges']) for x,y in pairs);b=sum(int(y['edges']) for x,y in pairs)
            ratios=[int(y['edges'])/int(x['edges']) if int(x['edges']) else None for x,y in pairs]
            relations.append(dict(readout=kind,family=family,label=rows[0]['label'],total_original=a,total_lcc=b,pooled_edge_retention=b/a if a else None,mean_pergraph_retention=avg(ratios),valid_nonzero_denominator_graphs=sum(v is not None for v in ratios),expected_graphs=16,lost=a-b,positive_original_but_zero_lcc=sum(int(x['edges'])>0 and int(y['edges'])==0 for x,y in pairs),original_zero_count=sum(int(x['edges'])==0 for x,y in pairs)))
    write_csv('relation_retention_summary.csv',relations)
    clean=read('lcc_minus_original.csv');ci=collections.defaultdict(list)
    for r in clean:ci[r['train_seed'],r['readout'],r['metric_group'],r['metric']].append(number(r['difference']))
    clean_seed=[dict(train_seed=int(s),cohort='old_two' if int(s) in SEEDS[:2] else 'new_six',readout=k,metric_group=g,metric=m,value=avg(v),valid_graphs=sum(x is not None for x in v),expected_graphs=2) for (s,k,g,m),v in ci.items()]
    write_csv('cleaning_seed_means.csv',clean_seed);clean_group=[];cg=collections.defaultdict(list)
    for r in clean_seed:cg[r['readout'],r['metric_group'],r['metric']].append(r)
    for (k,g,m),rows in cg.items():
        for label,seeds in [('old_two',SEEDS[:2]),('new_six',SEEDS[2:]),('all_eight',SEEDS)]:
            values=[r['value'] for r in rows if r['train_seed'] in seeds];a=avg(values)
            clean_group.append(dict(readout=k,metric_group=g,metric=m,cohort=label,mean_difference=a,median_difference=statistics.median(values) if a is not None else None,negative_count=sum(v<0 for v in values) if a is not None else None,zero_count=sum(v==0 for v in values) if a is not None else None,positive_count=sum(v>0 for v in values) if a is not None else None,expected_seeds=len(seeds),valid_seeds=sum(v is not None for v in values)))
    write_csv('cleaning_cohorts.csv',clean_group)
    def value(seed,kind,version,group,key):return number(next(r['value'] for r in cells if int(r['train_seed'])==seed and r['readout']==kind and r['version']==version and r['metric_group']==group and r['metric']==key))
    def aggregate(kind,version,group,key,seeds=SEEDS):return avg([value(s,kind,version,group,key) for s in seeds])
    for kind in KINDS:
        rr=[r for r in ret if r['readout']==kind];summary[kind]=dict(nodes_lcc_mean=avg([float(r['nodes_lcc']) for r in rr]),nodes_lcc_min=min(float(r['nodes_lcc']) for r in rr),nodes_lcc_max=max(float(r['nodes_lcc']) for r in rr),edge_retention_mean=avg([float(r['edges_retained']) for r in rr]),edge_retention_min=min(float(r['edges_retained']) for r in rr),mean_edges_lost=avg([float(r['edges_original'])-float(r['edges_lcc']) for r in rr]),type_retention={n:avg([float(r[n+'_retained']) for r in rr]) for n in ['Gene','Disease','Chemical','Species']},versions={})
        for version in ['original','lcc']:
            summary[kind]['versions'][version]={key:aggregate(kind,version,group,key) for group,key in [('raw','nodes'),('raw','edges'),('absolute_error','edges'),('distribution_error','main_incident4_W1'),('distribution_error','main_joint_degree_TV'),('absolute_error','triangles'),('distribution_error','triangle_composition_TV'),('distribution_error','MP2_composition_TV'),('distribution_error','MP3_composition_TV'),('raw','average_clustering'),('raw','cpl_finite_ordered_pairs')]}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    primary=[]
    for seed in SEEDS:
        primary.append([seed]+[value(seed,kind,v,'absolute_error','edges') for kind,v in [('raw','original'),('raw','lcc'),('ema_0.99','original'),('ema_0.99','lcc')]]+[value(seed,'ema_0.99','lcc','absolute_error','edges')-value(seed,'raw','lcc','absolute_error','edges')])
    details=['# 完整关键对照表','', '所有均值先在训练seed内平均两个样本，再等权汇总；不以总体均值掩盖逐seed方向。','',
             '## 保留量','',table(['读出','主体节点均值','最小','最大','边保留均值','边保留最小','平均失去边'],[[k]+[summary[k][v] for v in ['nodes_lcc_mean','nodes_lcc_min','nodes_lcc_max','edge_retention_mean','edge_retention_min','mean_edges_lost']] for k in KINDS]),'',
             '## 类型保留率','',table(['读出','Gene','Disease','Chemical','Species'],[[k]+[summary[k]['type_retention'][n] for n in ['Gene','Disease','Chemical','Species']] for k in KINDS]),'',
             '## 原始／主体版本结构指标','',table(['读出','版本','边数绝对误差','incident4 W1','联合度TV','三角绝对误差','三角组成TV','MP2组成TV','MP3组成TV'],[[k,v]+[summary[k]['versions'][v][key] for key in ['edges','main_incident4_W1','main_joint_degree_TV','triangles','triangle_composition_TV','MP2_composition_TV','MP3_composition_TV']] for k in KINDS for v in ['original','lcc']]),'',
             '注意上表edges和triangles列是逐图绝对误差的平均，不是原始数量；原始数量全在metrics_long.csv与relations.csv。','',
             '## 主比较各seed：平均绝对边数误差','',table(['训练seed','raw原图','raw主体','EMA0.99原图','EMA0.99主体','主体EMA−raw'],primary),'',
             '## 三衰减率全部分组主指标','',table(['读出','版本','分组','平均配对差','中位差','改善／持平／恶化'],[[r['readout'],r['version'],r['cohort'],number(r['mean_difference']),number(r['median_difference']),r['negative_count']+'/'+r['zero_count']+'/'+r['positive_count']] for r in cohort if r['metric_group']=='absolute_error' and r['metric']=='edges']),'',
             '## 样本差异','',table(['读出','版本','边Jaccard均值','边对称差均值','两样本度W1均值','MP2组成TV均值','完全相同对数'],[[k,v]+[avg([float(r[key]) for r in div if r['readout']==k and r['version']==v]) for key in ['labelled_edge_jaccard','labelled_edge_symmetric_difference','total_degree_W1_between_samples','MP2_composition_TV_between_samples']]+[sum(r['labelled_graphs_equal']=='True' for r in div if r['readout']==k and r['version']==v)] for k in KINDS for v in ['original','lcc']]),'',
             '全部32对在各自版本都有不同节点数或边数，故可由规模证明不是完全同构；这是这批有限样本的证据，不代表总体多样性充分，也不把节点重编号当多样性。','',
             '## 异常个例及缺失项','',
             '2026102845 / EMA0.9 / sample2026102821排除的7节点分量：7边、1个三角，节点类型1/4/1/1，最大度4。其余63图不损失三角，全部64图不损失四环。','',
             'Species保留最差个例为2026102844 / raw / sample2026102821，只保留6/12；Disease最差为2026102844 / EMA0.97 / sample2026102821，保留55/74。没有图完全缺失某类，但稀少类型低度节点损失需要披露。','',
             'HGEN原生powerlaw依赖缺失、可选社区标注缺失、完整orbit谱未实现；详见METRICS.md。当前项目固定xmin尾部拟合另名保留。无虚构值、无依赖安装、无新模型调用。']
    (ROOT/'FULL_TABLES.md').write_text('\n'.join(details)+'\n')
    text='''# 最大连通分量使用版：结论

**建议作为可选的“连通核心使用版”，与原始图共同保留并明确标注；不建议无说明地当作完整异质图分布的代表。** 本轮对全部64图生成了独立LCC派生图，没有筛样、补边或改模型；参考图确为全连通192节点、926边。

## 保留了什么

| 读出 | 主体平均节点 | 平均节点保留 | 平均边保留 | 平均排除边 |
|---|---:|---:|---:|---:|
| raw | 177.75 | 92.58% | 99.816% | 1.6875 |
| EMA0.9 | 178.00 | 92.71% | 99.793% | 2.0000 |
| EMA0.97 | 176.25 | 91.80% | 99.819% | 1.6875 |
| EMA0.99（主比较） | 175.44 | 91.37% | 99.849% | 1.5000 |

最大主体范围155—192节点；64图都保留四类型、没有最大分量并列。本批全部四环保留；只有一张EMA0.9图丢失一个三角。所有被排除分量的最大节点度不超过4，没有排除按参考各类型P99阈值定义的hub。简单MP2/MP3数量与组成变化很小：EMA0.99的MP3完全保留，MP2每图平均只损失0.1875条。

损失并非类型均匀：EMA0.99平均保留Gene95.31%、Disease86.99%、Chemical95.17%、Species84.38%。某张raw图只保留6/12个Species。少量绝对节点损失可以是稀少类型较大的比例损失，不能只看全图节点保留率。

没有任何“原图存在的关系族”因取主体而完全消失。按每种读出的16图汇总，全部关系族的边保留率至少98.50%；Species--Species原来有些图就没有边，取主体并未进一步损失该族。逐图0分母保留null；关系汇总的总边数比率明确标为pooled，不冒充只平均有效图。

## 主体保真与机械变化

取主体基本保留已有三角、四环和元路径，也基本保留原来的优点与错误。它不会解决原有过多三角或类型组成偏差。

删除低度外围后，度分布并没有自动更接近参考：raw主W1从0.8774升至0.9369，EMA0.99从0.5225升至0.5852。主口径固定参考类型权重，经验PMF用实际保留节点数归一；另报实际类型权重兼容值，结论方向相同。EMA0.99联合度TV约0.7835→0.7831，仍不小，不能说角色关系组合已准确。

平均聚类上升也有机械成分：EMA0.99约0.3682→0.4034，而参考为0.3365。它的三角全部保留、低度节点被移除，聚类均值的分母变了。连通率1、LCC占比1及零孤立点是处理规则的后果，不是模型新学到的能力。关系容量随真实节点集合重新计算；即使边几乎没丢，密度仍可能增加。

## EMA的收益是否仍在

主比较EMA0.99在主体口径的平均绝对边数误差为41.3125，raw为218.7500，平均配对差−177.4375、中位差−89.5，仍为6个seed改善、2个恶化。原始口径为219.5625→41.6875，方向基本不变。

新增六seed单列：主体raw89.0000→EMA38.9167，平均差−50.0833、中位差−51.5，4改善／2恶化；原两个开发seed为608.0000→48.5000，2个改善。0.9、0.97及全部逐seed结果完整附报，不因本次后处理改变0.99主比较。

主体口径下W1也仍是EMA较好（0.9369→0.5852），但不能用取LCC抹掉节点覆盖代价。原seed2026102833的EMA主体比raw少21.5节点，这一差异仍存在；只因二者LCC占比都变1，不能称其代价消失。2026102845的EMA数量／W1退化也未被清理修复。连通核心适用性与原模型改进证据应分开解释。

## 样本差异与使用限制

未观察到取主体后两样本变成同一图：每读出8个训练seed的两样本，原始及主体版本均不完全相同，且本批可由节点/边数不同证明非同构。编号对齐的边Jaccard仅轻微变化；结构分布的两样本差异也没有显示普遍坍缩。但每模型两图不能证明总体拓扑多样性充分，编号边差也不是同构不变的多样性指标。

若用途侧重连通主体上的传播、局部关系结构及路径分析，可以采用本派生版本；必须同步披露保留节点、各类型流失、关系容量／密度和来源映射。若用途依赖外围人口、稀少类型覆盖、低度分布或全体节点统计，应同时提供原图，不能只给LCC版本。没有以零孤立点或逐项匹配参考作为硬性通过标准，也没有给所有用途设任意综合分数。

## 交付和证据

- METRICS.md：完整指标清单、HGEN历史命名差异、变节点数适配及缺项；不把小表称完整原生HGEN复现。
- FULL_TABLES.md、retention.csv、types.csv、relations.csv：保留量、类型、关系、实际容量和密度。
- metrics_long.csv、seed_means.csv、ema_pairs.csv、cohort_pairs.csv、lcc_minus_original.csv：逐图原值/参考/绝对误差及两种配对、旧2/新6/全8、均值/中位数/方向数量、null原因。
- cleaning_seed_means.csv、cleaning_cohorts.csv：处理本身LCC−original也先按训练seed平均两图，再分旧2/新6/全8，保留中位数与方向；relation_retention_summary.csv另给关系保留及零分母覆盖。
- triangles.csv、closure.csv、paths.csv：全部类型三角、二跳机会/闭合、MP1/2/3及允许重复节点walk明细；完整四环类别和度排序谱在逐图JSON。
- graphs/：64份未改的原图副本、64份LCC派生图、编号映射及逐版本完整指标；excluded_components.json逐项记录被排除分量。
- protocol.json、metric_sources.json、vendor/：定义及源码哈希，输入清单绑定原a4a87a19结果；源码通过AST仅加载纯统计函数，没有导入模型。
- tests.xml、audit.json：13项新增必要CPU测试、64图映射/诱导关系/实际容量/路径分量可加性和汇总算术审计。原始64图冻结指标全部复现，未重跑历史O—E实验。

HGEN原生powerlaw依赖缺失，社区标签未提供、完整orbit谱不在本冻结实现内，明确列为缺项；已有固定xmin尾部拟合保留原定义，未冒称等价。全程CPU单线程；本轮训练、模型前向、采样和GPU使用均为0。完成后停止，不追加修图或科学实验。
'''
    (ROOT/'README.md').write_text(text)
    print(json.dumps(dict(status='report_complete',readouts=len(summary))))
if __name__=='__main__':main()
