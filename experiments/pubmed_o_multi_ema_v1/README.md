# PubMed O：八训练seed、三EMA尺度（准备版）

基线：`e42c962f03b83c183c17cee09d948c674f2e69b9`。本包只完成接线和工程验收，**没有正式运行授权，没有激活账本**。旧两seed／单EMA／130分钟方案暂停，不复用其审批。

## 固定科学合同

PubMed attempt02，192节点、926边，O基础网络；package内58份模型、数据、损失、初始化、normal2、后验及评价文件与基线逐文件相同。没有CN、ReFeX或RolX激活路径。完整18336槽查询，FP32，原AdamW及学习率，1000次normal2更新，T1000，不改生成规则。

八seed：2026102812、2026102833，以及2026102841—2026102846。本地可用项目记录中未找到新增六seed既往使用；查找范围和限制见seed_manifest.json。旧两seed参与过方案选择，始终与新增六seed分组报告。

旧两seed按initial_assets.json读取并验证封存的0更新科学权重；新建空AdamW。新六seed调用原package/experiment_contract.py的initialization：原anchor构造、独立模块初始化、实际公共参数复制；之后重置原配对Python／NumPy／Torch CPU／所选CUDA／显式Generator随机流。正式入口在首次前向前保存原始初态、全部EMA、空优化器、RNG和绑定。新增初态本票未提前训练或执行前向。

每条轨迹维护三个独立参数副本，β=0.9、0.97、0.99：

`bar_theta[0]=theta[0]`

`bar_theta[k]=beta*bar_theta[k-1]+(1-beta)*theta[k]`

第1步也递推，不直接复制theta1。每次完整normal2原参数更新后执行三次独立递推，来源均为同一raw，EMA不进优化器、不参与反向或训练输入。buffer随读出复制当前raw buffer，不做EMA。保存0／500／1000，包括raw、三个EMA及各自计数、优化器与RNG；各读出验证恢复raw与训练随机状态。只导出最终1000更新四个独立文件。

## 队列与结果

八卡一卡一seed；所有训练成功封存后，各卡依次执行raw、ema_0.9、ema_0.97、ema_0.99，每种依次采样2026102821／2026102822，共64条链。完整身份为训练seed＋读出＋采样seed。每链验证自己文件的哈希、衰减率、源绑定及最终检查点。原先验、完整查询和逐槽随机消费不变，汇总额外核对各读出的采样初态与RNG一致。队列见evidence/queue.json；dry-run不创建运行目录、不启动worker。

主比较仍为ema_0.99−raw；另两种同时完整展示，不事后挑赢家。每图先计算相对926的绝对边数误差，再按训练seed平均两图，分别汇总8个差值、等权均值、中位数及改善／持平／恶化数量；旧2／新6分别报告。跨seed有符号偏差均值、范围、总体标准差仅为次要描述。缺任一必需图不只平均成功项；各类结构指标与null保留。逐seed列出W1增加超过10%、LCC下降超过19.2的预警，不将它们当安全证明或全轮自动否决。两图不重复不等于总体多样性充分。

## 本票实测

- CPU40项通过、0失败、0跳过（旧测试及新增三EMA／队列／统计测试）。最初5项新增测试引用错了内部属性名，修正测试后完整重跑；失败过程保留在evidence/preparation_report.json。
- GPU0 RTX2080Ti：5次临时normal2更新，2次raw、2次三EMA、1次恢复后重复第2步。工程状态已按用户要求删除，仅保留哈希及核验回执，不进入科学初始化。
- 本段raw对照与三EMA：输入、训练RNG、模型、梯度、优化器逐位一致；加载模型／优化器／三EMA／RNG／计数精确一致；恢复后续算逐位一致。四种验证后raw、模式、优化器及RNG恢复。
- 暖normal2含三EMA3.5585秒，其中原更新3.2592秒、三EMA约0.2993秒；一次完整四读出固定验证7.8948秒。检查点写入2.856秒，恢复1.096秒。
- 三份shadow79,591,500字节，实测新增allocated75.99MiB。峰allocated3267.69MiB，reserved3638MiB，CPU RSS4823.38MiB（含取证副本）。这是单卡短段，不是八卡稳定吞吐或完整链成本。
- package和科学数据未改；正式训练0、完整链0、新图0。没有八seed GPU复测，没有新跨CPU/GPU等价要求。旧证据不改写。

## 正式预算建议，尚待确认

固定八卡队列：训练关键路径约60.36分钟；八轮链历史最大621.82秒／条，合计82.91分钟。再计初始化／验证／保存约1.49分钟，计算关键路径144.77分钟；10%未实测并发／状态／进程启动余量14.48分钟；评价、审计、导出及交付20分钟，总规划179.24分钟。建议批准**185分钟**，不是完成保证，也不是八条链只算一轮。

三EMA固定验证、写盘和恢复已计入；链成本沿用前两轮O的实测，不新跑完整链。8个同模型任务的CPU／IO竞争、EMA自由生成状态成本仍未实测。实际正式启动须重新核实8卡权限、他人进程、显存和至少12GiB输出卷空闲；少于8卡只报告，不自动缩规模。初态／检查点约3.6GiB，加读出约0.8GiB及日志状态，12GiB是保守存储门，不代表持久性认证。

`initial_assets.json`和`prior_evidence/`保留原两seed历史记录（含130分钟旧提案字段），仅作来源证据；本版有效提案为protocol.json与evidence/cost.json。

## 命令

使用已有sparse_block环境，不安装升级依赖：

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CUDA_VISIBLE_DEVICES='' PYTEST_ADDOPTS='' python -m pytest -q test_ema.py test_runtime.py test_multi_ema.py
CUDA_VISIBLE_DEVICES='' python run.py dry-run --output /绝对路径/尚不存在的新运行目录
```

已完成GPU预检不用重复运行。若迁移后必要证据路径变化，只重建cost文件中的证据路径绑定，不改科学设置：

```bash
CUDA_VISIBLE_DEVICES='' python build_cost.py --preflight evidence/gpu_preflight/preflight.json --output /绝对路径/新的cost.json
```

以下正式命令**尚未授权执行**；本票不创建审批文件。将来经审查及用户确认185分钟或新预算后，按runtime_contract.approval_gate要求登记真实授权、最终源码／协议／cost哈希、实际8卡UUID及资源归属、预检environment、旧初态路径、全新输出目录。不能把文档或模板当真实审批。预算变更须明确确认并更新绑定，不能私改截止。

```bash
python run.py formal --output "$NEW_RUN" --approval "$CONFIRMED_APPROVAL" --cost "$BOUND_COST" --gpu-uuids "$GPU0" "$GPU1" "$GPU2" "$GPU3" "$GPU4" "$GPU5" "$GPU6" "$GPU7"
```

唯一入口run.py通过管道票据启动worker；不直接调用worker。STOP为在NEW_RUN创建STOP文件，已运行任务按边界停止并保留证据；无自动重启、补抽或延时。末20分钟用于停止、统一评价与交付。SIGKILL不能捕获。停止／失败结果汇总保留缺图及原错误。

全部检查支持实现接通，不证明EMA改善生成。参数平滑可能保留持续偏差，也同时引入滞后；需正式结果回答准确性、收益覆盖和结构代价。
