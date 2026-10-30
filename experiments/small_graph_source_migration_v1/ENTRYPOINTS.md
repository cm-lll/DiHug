# 入口地图（保留历史源码，不自动执行）

|功能|路径|实际用途与限制|
|---|---|---|
|A/C模型|snapshots/ac_screen/package/model_variants.py|StructuralDenoiser，A有CN、C无CN；真实字典由model_io读取|
|A/C初态与训练|snapshots/ac_screen/model_io.py、run_worker.py|历史3000更新、EMA0.99；四组筛选队列仍保留|
|A/C总入口|snapshots/ac_screen/run.py|`dry-run`或`formal`位置参数；formal需新审批，原GPU/资源/版本检查仍有效|
|E-earlyCN注入|snapshots/early_count_length/package/early_cn.py、hsgd/model.py|37→64边残差及末端头裁列；精确以源码为准|
|CE/有界数量项|snapshots/early_count_length/package/count_aux.py、normal2.py|λ=0.001由protocol/model_io绑定；CE臂权重0|
|0到500训练入口|snapshots/early_count_stage500/run.py、run_worker.py|依赖未打包的封存0步.pt，不能原样运行|
|500之后训练/阶段采样|snapshots/early_count_length/run.py、run_worker.py、resume_io.py、stage.py|完整恢复、阶段隔离；依赖未打包父checkpoint与旧账本|
|IMDB续到6000|snapshots/imdb_continue/launch.py、continue_worker.py|历史路径和期限写定；只供迁移参考|
|EMA|各快照ema.py|从θ0递推、更新计数、buffer复制、保存恢复|
|噪声/后验/查询|各快照package/hsgd/kernel.py、runtime.py、noising.py|原支持与后验；完整图分块未重新接线|
|原图/LCC完整指标|evaluation/frozen_evaluation_source/lcc_eval/metrics.py|make_graph/types+2×E+schema；stats/compare；不直接运行历史vendor脚本|
|指标展开|evaluation/frozen_evaluation_source/evaluation_fields.py|scalar(stats,reference_stats,compare,type_names)输出raw和distribution_error|
|最终结果重算入口|evaluation/source/recompute_cpu.py|`--root 已交付结果目录 --output 新目录`；需要原结果evaluation.json/reference/run布局，本源码包不含结果|
|最终汇总|evaluation/source/build.py、report.py|可信任务身份修复及全部字段，含旧服务器路径，需迁移结果读取层|
|发布错误修复|evaluation/corrected_length_finalize.py|相对历史执行finalize的报告修复，不是新的科学训练实现|

核对本包（可直接执行，不导入模型）：

```sh
python -B verify_bundle.py
```

已经另行持有最终结果包时，以下原有CPU入口可用于该**结果包**，不是对源码目录误运行：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B evaluation/source/recompute_cpu.py \
  --root /新位置/已解压的完整结果包 --output /新位置/独立CPU重算目录
```

不要执行package/prepare.py、select_dictionary.py或refit_dictionary.py来“准备”现有字典；它们为历史材料，正式资产已冻结。本包没有要求或执行拟合。
