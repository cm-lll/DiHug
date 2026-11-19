"""Produce reviewable result tables and factual manuscript inserts only after acceptance."""
from common import *
import csv
def f(x):return 'NA' if x is None else f'{x:.6g}'
def main():
    assert read(RUN/'training_acceptance.json')['passed'] and read(RUN/'generation_acceptance.json')['passed']
    s=read(RUN/'summaries.json');a=read(RUN/'generation_acceptance.json');out=RUN/'writing';out.mkdir(exist_ok=False)
    rows=[]
    for g in s['groups']:
        row={k:g[k] for k in ['dataset','kind','arm','unique_outputs','training_repeats']};row['available_starts']=len(g['available_starts'])
        for k,v in g['metrics'].items():
            for stat,x in v.items():row[k+'_'+stat]=x
        rows.append(row)
    with (out/'main_and_supplement_tables.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# 新增固定面板：已验收结果','',f"96 次正式拟合，451200 次更新；{a['full']} 张完整流程终图，{a['suffix']} 张共享阶段二终图。N 没有训练重复；生成图不是独立训练次数。",'',
        '完整流程与共享阶段二分开；旧实验保留，不与本轮统一混合训练配方拼表。主误差为类型三角向量和跨类型四环向量的相对 L1 误差等权平均，不能与旧两模体误差直接相减。','',
        '|参考|比较范围|条件|可用起点/8|输出数|主要误差均值|输出SD|训练seed均值SD|进入阶段二比例|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for g in s['groups']:
        v=g['metrics']['primary_error'];lines.append(f"|{g['dataset']}|{g['kind']}|{g['arm']}|{len(g['available_starts'])}|{g['unique_outputs']}|{f(v['mean'])}|{f(v['output_sd'])}|{f(v['training_seed_mean_sd'])}|{f(g['metrics']['stage2_reached']['mean'])}|")
    lines+=['','主要配对差为对照−M，正值有利于M；下面区间仅在同一参考、同一比较范围的三个对照中校正，不是全部数据集统一显著性。','']
    english=[r'\subsection{Additional fixed capacity panel}',
        'We compared the original three-hidden-layer scorer (M), a small one-hidden-layer scorer (F), a parameter-matched wide one-hidden-layer scorer (W), and a no-learning control (N). All learned scorers received the complete original edit features and the same mixed micro-corruption/prior-bridge training cache. Each learned condition used eight independent training seeds and eight shared generation starts. The no-learning control has eight unique outputs per reference, not eight training repeats.',
        'The exact historical IMDB and PubMed references were retained; Freebase and full-relation HAN ACM each supplied a fixed 2,048-node induced reference selected before model outcomes. Each reference was fitted separately. The binary undirected views and their different relation inventories describe the evaluation scope; they do not establish transfer across networks or a universal ordering of complexity.',
        'Full-pipeline comparisons and shared-stage-two comparisons estimate different effects. For the latter, the prefix producer was fixed to M with training seed 810 before outcomes. We retained every naturally reached prefix, including its remaining attempt budget, and never replaced a failed start. The resulting intervals are conditional on this producer and do not include its fitting uncertainty.',
        'The primary error is the equally weighted mean of relative L1 errors for the full schema-allowed typed-triangle count vector and the alternating cross-relation four-cycle count vector. Zero-reference categories are retained. Each vector is normalized by its total reference count, floored at one. This endpoint is reported separately from historical two-motif errors.',
        r'\begin{center}\begin{tabular}{lllrrr}\hline Reference & Scope & Arm & Starts & Mean & Output SD\\\hline']
    for g in s['groups']:
        v=g['metrics']['primary_error'];english.append(f"{g['dataset']} & {g['kind']} & {g['arm']} & {len(g['available_starts'])}/8 & {f(v['mean'])} & {f(v['output_sd'])} \\\\")
    english += [r'\hline\end{tabular}\end{center}',
        'Training-seed mean standard deviations and all per-output observations are supplied in the accompanying tables. Generated outputs are not counted as independent model fits. Crossed training-seed/start bootstrap intervals use 20,000 draws and 98.333 percent endpoints for a nominal 95 percent three-contrast family within each reference and estimand. These are separate families, not an experiment-wide superiority test.']
    for z in s['contrasts']:
        ci=z['interval_98_333_percent'];interval='不可估计（起点不足）' if ci is None else '['+', '.join(f(v) for v in ci)+']'
        verdict='本比较族内支持M主要指标优势' if z['M_advantage_established_in_this_family'] else '未建立M主要指标优势；也不代表等价'
        lines.append(f"- {z['dataset']} / {z['kind']} / {z['contrast']}: {f(z['mean'])}，98.333%区间{interval}；{verdict}。")
        if ci is not None:english.append(f"For {z['dataset']} ({z['kind']}), the {z['contrast']} paired primary-error difference was {f(z['mean'])}, with interval [{f(ci[0])}, {f(ci[1])}].")
    missing=[]
    for ds in ['imdb','pubmed','freebase','acm']:
        n=sum(v['available'] for k,v in a['prefix_availability'].items() if k.startswith(ds+'/'))
        if n==0:missing.append(ds);english.append(f"No eligible shared-stage-two prefix was reached for {ds}; its stage-two effect is unestimable under this fixed protocol.")
    lines += ['', '所有未到达均保留。阶段二起点由固定M810产生，条件化区间不能代表任意前缀或跨网络泛化。不能从显著性缺失推出等价，也不能由少数参考的差异证明复杂网络普遍不可替代。',
        'Delicious旧失败及晚期平台现象仍保留；没有新增Delicious训练或延长轨迹。HGEN版本资格和旧阴性结果保持。',
        '验收范围：CPU独立端点指标、来源绑定、不变量及训练保存概率核查；不是每个GPU候选步骤的全轨迹重放。',
        '本文稿插入段落与表格已生成；最终整稿整合、摘要论断和版式仍须依据这些结果复核。']
    english += ['These observations delimit empirical capacity effects under a fixed optimizer and candidate policy. Parameter matching does not equate computation or optimization. We make no universal irreplacability, equivalence, exact reverse-diffusion, convergence, or cross-network generalization claim.']
    (out/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n');(out/'MANUSCRIPT_INSERT.tex').write_text('\n\n'.join(english)+'\n')
    response='We added a fixed four-reference capacity panel with three full-feature learned scorers and a no-learning control. The wide shallow scorer approximately matches the original scorer\'s parameter count, separating small-model comparisons from a parameter-budget comparison. Each learned condition has eight training seeds, with shared generation starts and explicit uncertainty. We report full-pipeline and conditional shared-stage-two outcomes separately, retaining all failed arrivals. Results, including unfavorable contrasts, are in the added table and the per-seed supplement. These experiments assess conditional empirical capacity benefits; they do not prove that the original architecture is universally necessary, nor that independently fitted references demonstrate transfer.\n'
    (out/'REVIEWER_RESPONSE_INSERT.txt').write_text(response)
    put(RUN/'writing_receipt.json',{'completed':True,'outputs':[str(p) for p in out.iterdir()],'whole_manuscript_integration':'pending author/agent review of accepted results','no_new_experiments':True})
if __name__=='__main__':setup();main()
