"""Write the descriptive report from all frozen final outputs. No model calls."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'experiments/role_condition_coverage_v1'))
from common import RESULT,read

def main():
    assert read(RESULT/'controller.json')['status']=='completed'
    cond=read(RESULT/'conditions.json');mac=read(RESULT/'macro_summary.json')['conditions']
    per=read(RESULT/'per_seed.json')['summaries'];cfg=read(RESULT/'resolved_config.json')
    def get(arm,name,key):return next(x for x in mac if x['arm']==arm and x['condition']==name)[key]['mean']
    def pair(name,key,d=3):return f'{get("narrow",name,key):.{d}f} → {get("broad",name,key):.{d}f}'
    lines=['# 扩大条件及合法干净图覆盖：完整CPU配对结果','',
      '**结论：覆盖扩大明显改善了本轮六个新测试目标的整体离散度分布，但不是完整数值控制的解决证明。**',
      '三个训练seed在全部三组合、三数值目标上的Genre W1均下降。正控制也改善；原极端目标、部分连通率、D–M边数有代价。',
      '0和11这两个新度值自身的精确频数没有改善，必须与“包含新值的整个目标W1改善”区分。',
      '', '## 冻结与执行', '',
      '科学源795b4bafb3259dc00f67cfb008991fb1fe5a1124，基线fdc446ca2a3f33ec5ae89b167d659559d2662641。',
      '唯一10800秒账本从北京时间2026-10-21 06:44:34.002开始，截止09:44:34.002，包含首次准备、失败修复、检查、短测、科学计算和交付。没有重置。',
      '六CPU单线程作业；两臂×三原训练seed2026102011/12/13×6000更新，实际36000更新全部完成。',
      '每模型11目标×8配对seed2026102031…38，实际528条全新T64/R4链全部完成；33792反向步、135168采样网络前向。',
      '18个1000/3000/6000完整模型/优化器/RNG检查点，实际初始和最终权重均保留。',
      '两臂实际初始482张量完全一致，来自原capacity_C_initial.pt；独立新AdamW连续训练，没有从旧6000或短测权重续训。',
      '8层256/64/128、8头、FF256/64/256、CN32/联合头128，实际6658481参数。原FP32网络、float64核、稳定logits CE、每微批sum*R/S、两普通输入各1/2，',
      'AdamW lr2e-4/wd1e-12/betas(.9,.999)/eps1e-8/AMSGradFalse，无裁剪/调度器/额外损失/人工错边/采样控制/硬角色/修图。',
      '原精确DP200000及CN路径2m门不变；CPU普通模式、线程1、highest/TF32False，没有GPU、近似读取、大图或notebook/BW。',
      '', '## 目标、来源和划分', '',
      '先固定训练度值集合1…18去除7/11，留出0/7/11，再由可行整数分区的固定规则选目标，未用生成效果挑选。完整规则见PROTOCOL/conditions.json。',
      '3个旧开发目标均未进入训练；旧moderate本轮不生成，旧single/double仅作已查看过的开发对照。',
      '组合测试只用广训练中出现的度值，但完整多重集未见；数值测试0是低于训练范围的外推，7/11是范围内插值。',
      '只改Genre；6个Genre、MG24边、Movie侧12个度1/6个度2，D/A/M完整联合向量频数、原schema和DA/DM/AM10/18/36不变。',
      '所有20个条件均通过整型流、二值简单图和完整联合统计可实现性核对。目标工件只有匿名向量/频数，不包含节点角色对应或真实边。',
      '', '|训练条件|Genre目标|', '|---|---|']
    lines += [f'|{n}|{cond["vectors"][n]}|' for n in cond['training']['broad']]
    lines += ['', '离线干净图采用相同固定DMA骨架、seeded行列顺序的二部整型流、160次尝试合法2switch；不声称均匀图采样。',
      '各臂48源样本：窄臂2×24，广臂12×4，原两条件首4源图逐张共享。每条件至少4个不同训练拓扑，验证每条件2图，共24图，与训练做完整类型/关系保持同构排除。',
      '窄平衡24源图为24拓扑；窄极端24源图为12拓扑（频数1,4,3,2,2,3,2,1,3,1,1,1），广臂每条件4图均不同。',
      '不同条件的Genre多重集不同，不能同构。类型内重编号只在划分后增强；不将其算新拓扑。',
      '因此干预是“条件及对应合法训练图覆盖”；不是只改一个数字的纯因果实验，实际不重复源拓扑数也不同。',
      '构造图和证明工具只用于合法监督/可行性，生成从原先验开始，推理case不含clean/clean_ids，不使用构造图作初始化或补救。',
      '', '## 完整最终离散结果', '',
      '每格先汇总同一训练seed的8图，再等权1/3汇总三个seed。W1是实际整数度经验分布的未缩放距离，不是预测期望度。',
      'MG偏差为实际边数减24。有符号排序度偏差、完整PMF/联合频数、均值/中位数/线性P90及8原值全部保留；不要求逐图匹配或W1为0。',
      '', '|组别/目标|向量|Genre W1 窄→广|MG偏差 窄→广|最大度 窄→广|次大度 窄→广|', '|---|---|---:|---:|---:|---:|']
    cats={'original_controls':'原控制','seen_development':'旧开发','training_middle_control':'训练正控制','unseen_combinations':'新组合','unseen_values':'新数值'}
    for name in cond['final_targets']:
        cat=next(x for x in mac if x['condition']==name)['category']
        lines.append(f'|{cats[cat]} {name}|{cond["vectors"][name]}|{pair(name,"G_W1")}|{pair(name,"family8_signed_error")}|{pair(name,"G_max",2)}|{pair(name,"G_second_max",2)}|')
    lines += ['', '训练正控制train_06由新增10条件的浓度下中位规则预选，12/5/4/1/1/1：W1 2.368→1.049，MG偏差+12.125→+1.792，max17.833→13.375，对目标12。',
      '说明学习到中间目标形态，但仍有校准偏差；不把“训练见过”自动当作充分学好。',
      '双hub开发目标：窄臂24/24图都是2个≥12节点；广臂为0个8图、1个12图、2个4图，24/24仍有2个≥8节点。',
      '最大/次大17.417/16.833→12.042/10.000，对目标10/10，有实质改进但最大度仍偏高。',
      '', '### 逐训练seed（不是将所有采样图当训练重复）', '',
      '|目标|seed11 W1窄→广|seed12 W1窄→广|seed13 W1窄→广|', '|---|---:|---:|---:|']
    for name in cond['final_targets']:
        cells=[]
        for seed in cfg['train_seeds']:
            a=next(x for x in per if (x['arm'],x['seed'],x['condition'])==('narrow',seed,name))
            b=next(x for x in per if (x['arm'],x['seed'],x['condition'])==('broad',seed,name))
            cells.append(f'{a["statistics"]["G_W1"]["mean"]:.3f} → {b["statistics"]["G_W1"]["mean"]:.3f}')
        lines.append('|'+name+'|'+'|'.join(cells)+'|')
    lines += ['', '六个新目标在三个训练seed下均改善整体W1；这是三个配对训练重复中的一致方向，不是18个独立训练重复。',
      '', '### 未见数值本身：频数证据混合', '',
      '每种测试目标都要求对应新值出现1个Genre。实际平均频数：', '',
      '|新值|窄|广|解释|', '|---|---:|---:|---|']
    for name,d,explain in [('value_0',0,'整体W1改善，但零度频数更偏离1'),('value_7',7,'频数更接近1，仍不足'),('value_11',11,'精确11度频数未改善')]:
        vals=[sum(s['G_mean_histogram'][d] for s in per if s['arm']==a and s['condition']==name)/3 for a in cfg['arms']]
        lines.append(f'|{d}|{vals[0]:.3f}|{vals[1]:.3f}|{explain}|')
    lines += ['', '因此只能说含未见值的目标整体分布泛化有进展，不能声称任意新度值已被准确控制。特别是0外推只有一个预定目标，不能外推一般数值外推能力。',
      '', '## 结构、频数和多样性代价', '',
      '|目标|Genre联合TV窄→广|Movie联合TV窄→广|Movie MG-W1窄→广|连通率窄→广|孤立点窄→广|DMA路径闭合窄→广|',
      '|---|---:|---:|---:|---:|---:|---:|']
    for name in cond['final_targets']:
        lines.append(f'|{name}|{pair(name,"G_joint_TV")}|{pair(name,"Movie_joint_TV")}|{pair(name,"Movie_relation8_W1")}|{pair(name,"connected")}|{pair(name,"isolates")}|{pair(name,"closure_path")}|')
    lines += ['', '原极端目标W1 .465→.611（seed11/12退化、13略好）。combo_1/2连通率70.83%→58.33%、79.17%→62.50%，',
      'LCC分别33.71→33.50、33.79→33.58；孤立点仅描述，不是硬否决，也没有删除。combo_3等另有连通改善，不能给统一无代价结论。',
      '广臂D–M平均边数偏差约−1.38…−1.79，而窄臂约−.08…−.50，对已知目标18存在更强欠连接；DA一般更近目标10，AM变化混合。',
      'D/A/M完整联合TV仍偏高且并非一致改善；如双hub开发目标Movie TV .882→.914。Movie MG边际W1多有明显改善，不能以此替代完整联合保真。',
      'DMA路径闭合约.388… .402→.396… .412，三角形通常略少；新条件没有唯一拓扑真值，不硬套旧闭合/三角形作为必须达到的数值。',
      'MP1/2/3完整类型-关系模式及总量已独立重算；细表/分子分母见原始generation及每图metrics，不以一个闭合率代替结构。',
      '66个模型/目标格，每格8图均为8个精确类型与关系保持同构类，频数全为1、观察有效类数8。没有去重后计算质量。',
      '这排除本批完全复制，不能证明总体多样性充分、稳定性或质量相近时的多样性无代价。不同条件/模型的类数不相加。',
      '', '## 学习机会与优化趋势', '',
      '各臂内部均匀抽本臂条件；实际窄臂每条件约2961…3039更新，广臂每条件467…548更新（每更新2普通输入，样本次数为2倍）。',
      '每条件训练机会约降到1/6，是本设计的明确代价。配对共享选择uniform、时间、查询和增强/噪声seed来源；条件和干净图不同，不声称带噪输入逐张一致。',
      '', '|seed|窄固定CE 3000→6000|广固定CE 3000→6000|', '|---|---:|---:|']
    for seed in cfg['train_seeds']:
        z=[]
        for arm in cfg['arms']:
            p=RESULT/'jobs'/f'{arm}_{seed}';a=read(p/'validation_u003000.json')['mean_ce'];b=read(p/'validation_u006000.json')['mean_ce'];z.append(f'{a:.6f} → {b:.6f}')
        lines.append(f'|{str(seed)[-2:]}|'+'|'.join(z)+'|')
    lines += ['', '固定验证是同一12条件、24独立拓扑×t1/32/64×4全覆盖，eval/no_grad且恢复全部训练RNG；不含六个独立测试。',
      'seed11广臂和seed13窄臂末段明显退化；其他也不能凭少量点宣称收敛。本票仍按协议使用全部final6000，没有挑3000或最好权重。',
      '训练loss平均的是不同条件混合，不能把两条训练曲线直接当相同目标分布下的比较；匹配固定验证和实际生成另报。',
      '', '## 正确性、成本与失败记录', '',
      '三原normal2@6000 SHA/环境核对，原固定12状态验证全部最大差0。20条件×3模型实际角色编码向量/频数精确一致，解析度似然95个端点/中间检查，',
      '软归属行和/列容量及类型内置换均通过；未发现机制计算错误，没有将泛化差当作实现失败。',
      '33项CPU回归零失败/跳过（原容量读取/连续保存恢复/STOP及新增覆盖接口）；独立解包复验另见unpacked_acceptance。',
      '准备阶段曾停下修正额外施加的“24极端图必须全不同”不可能门（固定骨架至多C(6,2)=15匿名MG选择）；改为先4不同、后记录训练重复，验证仍严格排除。',
      '另修正冻结目标JSON tuple/list比较和测试模块evaluate命名冲突，均在正式学习前完成。原失败日志/STOP收据存档，条件列表/原激活账本未变。',
      '正式训练、生成和独立审计没有失败、OOM或资源门超限，没有筛图、补抽、缩链、改参数或另开窗口。',
      '成本门使用全部训练条件、两臂/3seed、全部最终目标的完整模型；最慢训练P90 .613664s，dense完整扫步 .297942s，',
      '预计7076.96秒≤剩余80%=7847.69秒，含1.2系数和360秒审计交付余量，才自动学习。126次临时双输入工程更新的权重隔离，未进入正式初始化。',
      '实际训练最长3282.27秒，采样最长1099.96秒，完整独立审计40.34秒；到审计完成累计5421.36秒（90.36分钟），仍使用原3h账本。',
      '12个正式train/sample作业累计CPU25831.76秒（7.18CPU小时，6进程并行），单作业峰RSS770.07MiB；父审计及整体同时峰未采集，不能冒称整组峰值。',
      '原始检查点、优化器、初始/最终权重完整保存在本地持久results，发布库存哈希/大小/用途，不把私有PT文件推到GitHub。',
      'Python3.9.23/Torch2.1.0/PyG2.3.1/NumPy1.26.4/SciPy1.11.0/NetworkX2.8.7/pytest7.4.4，CPU线程1；环境全文见correctness及freeze。',
      '', '## 决策与停止', '',
      '1. 同目标广覆盖优于窄覆盖：六个独立目标整体W1在三个训练seed中方向一致，支持这个限定的数据覆盖干预。',
      '2. 中间正控制明显学到形态，仍有MG/max/完整频数残差，不宣称充分收敛或完全校准。',
      '3. 已见度值的新组合有明确有限面板进展；不外推任意联合分布或真实大图。',
      '4. 含未见值的目标整体改善，但精确新值频数恢复混合，数值泛化结论需限定；0、11不可包装成完全成功。',
      '5. 结构代价存在：极端控制、D–M欠连、两组合连通率及部分联合TV退化；未观察批内同构复制增加，但样本数有限。',
      '保留覆盖路线作为值得后续审查的方向，不把它升级为已解决所有角色控制问题的默认方案。此次完成后停止，不延训、追加候选、改学习率、近似读取或启动大图。']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('wrote RESULTS.md',len(lines),'lines')
if __name__=='__main__':main()
