# 本分支确认状态

2026-10-29 新增六任务准备已通过定向检查，独立目录 experiments/typed_structure_early_cn_expansion_v1（review worktree typed_structure_early_cn_expansion_review）。新增seed IMDB2026102931、PubMed2026102932；原seed D早融合、新seed E早/晚各两数据集，共拟6训练12链。D GPU工程各3次normal2、总6次；CPU16项通过，新seed早晚初态/完整随机流/训练与采样输入合同一致，原seed D/E公共初态一致。只把已审early install扩展到真实D，训练/采样循环及字典/核/EMA/指标未改。建议独立280分钟，尚未授权，不存在新增formal审批/起止/科学更新或完整链。原dbe96f5 GPU0/1两任务仍运行，源码与审批未改，原15:46:47/16:06:47截止不变。准备11:54:11起、12:39:11硬截止，证据results/typed_structure_early_cn_expansion_prepare_v1；发布回执另记。

2026-10-29 E-earlyCN独立准备：源472e533/result aa4227，独立experiments/typed_structure_early_cn_v1；CN37→64零初始化边残差在lin_in_E后/主首层前注入，末端CN移除，净少2368参数。CPU10项通过；IMDB/PubMed各3次临时normal2（总6），零投影对照、非零梯度、对称性/类型内重编号、完整模型/优化器/EMA/RNG恢复与续算均通过。科学训练0、完整生成0，GPU已释放。原Dropout草稿与停止回执保留且未合并。拟两卡2模型/4链3000更新，规划255.71分钟，建议正式预算260分钟，未授权。完整图暂停；等待独立源码审查。准备票11:00:10起、11:40:10截止，不重置。

2026-10-29 aa4227后的C→E固定权重定位：PubMed8次完整查询、0反向/更新/新图；普通t500净变化差+0.001013，两个封存X499差−0.057228/−0.074274。E无CN现有边更易删除，但末图GD/DC缺口不能由单时刻完整解释。CN已有32学习均值+4log1p计数+1存在标记，未发现本票可确认的错位；未捕获学习向量尺度，不额外前向。成果experiments/typed_structure_cn_diagnosis_v1。唯一待审候选为10%训练期CN整段dropout，未实现，未授权训练。完整图暂停，原图主分析、LCC独立，旧记录不改。下方旧准备状态为历史记录。

2026-10-29：完成最后一轮192节点A/C/D/E架构筛选的准备审查包。冻结3000次normal2、EMA0.99、八模型/十六条T1000链的运行与评价接线，但未授权、未启动正式任务。

八格各2次临时normal2，共16次；新增GPU工程检查通过。CPU当前21项通过、零跳过；独立ZIP解包179份载荷哈希通过，21项再次通过。78份模型、EMA、字典、指标来源文件与固定Git对象一致。工程初态及检查点本地保留、哈希公开，不进入正式初始化。旧CPU/GPU等价失败不改写。

实际完整数据静态核对：IMDB1653/4079、PubMed1565/12284；没有完整图模型加载、前向或字典拟合。完整图RolX字典及候选的资源短测仍待授权。

估算八卡计算主体201.53分钟，含评价、规划余量与交付合计271.76分钟；建议未来正式预算280分钟。当前没有可激活审批，没有正式起止时间。准备起点02:42:42、固定截止03:27:42，Asia/Shanghai。发布后停止。

详见experiments/typed_structure_final_screen_v1/README.md、FULL_GRAPH_READINESS.md及evidence。历史主工作区记录保持原样。
