# v1.1：只修事务、设备迁移与短测时限

基线0cdd8274d7359c4cc189685735fbe77c60c712c7。模型、损失、两份数据及完整科学配置
JSON逐字保留；仍为12000更新、千步永久保存、3000/6000/9000/10000/12000五评价点。
旧版本及ZIP不改。本包不运行训练或完整链来验证这些工程变化。

## 检查点事务

写入顺序：私有prepared载荷→包含完整内容摘要和实际字节SHA256的intent→
发布不可变checkpoint→latest→milestones。恢复先核对所有事务、载荷及已有指针，
再修缺失项；冲突不覆盖。校验覆盖模型、优化器、全部所存RNG、绑定、计数及extra，
不是只验模型权重。有效已有文件在内容完全一致时幂等接纳，不重新序列化覆盖。
prepared、文件已发布但latest缺失、latest完成但milestone缺失均可恢复；
u0在冻结面板落盘之前中断，按原已绑定seed/config确定性重建，非零更新缺面板仍拒绝。

没有intent且没有独立可信收据的孤立文件，单凭自算哈希不能在load时认证。
只有save调用方持有完整已知内存快照、与全部载荷逐项摘要一致时才可接纳。
这不是绕过v1到v1.1的源码绑定检查。正式续训只接受同一新版发行绑定。
持久化使用原子rename及fsync；仍不能承诺所有平台故障/磁盘损坏都可恢复，冲突保留留证。

## 显式设备迁移

默认严格；另加`--allow-device-migration`才允许UUID/index/visibility和设备名称改变。
源码提交、完整release manifest、config、data、run_id、backend、Python、Torch实际版本、
HIP/CUDA构建、数值依赖版本、执行模式均必须相同。该选项不允许不同源码版本续训，
也不能用于fresh-start。原binding.json和旧权重不改写；迁移回执单独保存在migrations/。
新检查点extra记录本次实际execution_binding，后续在该设备恢复可继续默认严格检查。

```bash
bash run.sh pipeline --dataset IMDB --device-index 0 --resume \
  --output /root/private_data/joint_cn_rounds_v1_1_run01/IMDB_seed2026101901 \
  --accept-config-provenance-gap --allow-device-migration

bash run.sh sample-checkpoint --dataset IMDB --device-index 0 --resume \
  --run /root/private_data/joint_cn_rounds_v1_1_run01/IMDB_seed2026101901 \
  --update 6000 --output /root/private_data/joint_cn_rounds_v1_1_sample01 \
  --allow-device-migration
```

第二条用于已存在的采样状态；第一次指定检查点采样去掉--resume。
跨设备读取仍保存在旧设备上的源检查点，每次都须显式给迁移选项。
恢复CPU、Python、NumPy及**所选卡**RNG；不要求恢复后跨设备轨迹逐位相同。
新入口不调用get_rng_state_all/set_rng_state_all/manual_seed_all，也不改变可见性变量。
为保持核心文件不改，只在同步reverse_step调用期间把原回滚RNG帮助函数接到所选卡实现，
随后恢复原帮助函数。这是单工作线程接口，不声称线程并发安全。多卡实机验证尚未执行。

## 570秒覆盖范围

`dual.cli`现在是标准库轻入口。数值库导入、数据核验/载入、CUDA初始化、取得设备锁、
全部计算、worker保存及父进程完成证据核验均在同一570秒期限内。
结束后单独记录cleanup_seconds及cleanup_budget_seconds=3；只终止本监督器启动的进程组。
SIGKILL不可捕获；超时/STOP/信号或证据缺失均非零，不将部分产物标为完成。

```bash
bash run.sh short-test --dataset PubMed --device-index 0 \
  --output /root/private_data/joint_cn_rounds_v1_1_preflight/PubMed_check01 \
  --max-seconds 570
```

本地超时测试用合成小进程模拟加载/CUDA/锁阶段；设备RNG迁移测试用明确的API mock
及CPU小图，不冒称真实BW兼容、显存或跨卡恢复已经通过。

## 来源补充

用户只读审查提供旧正式binding收据，预期SHA256为
95b8c32c5ab649ab017d5ff2ef2a975a14b2b97893e5164b2fb5658e3bf1a513。
旧seed2026101407；AdamW lr0.0002/weight_decay1e-12/amsgradFalse，无裁剪/调度器。
本地没有该文件字节及旧构造源码，因此沿用用户历史收据，不声称重新认证远端文件。
betas/eps仍未独立核实；当前配置不变。详情data/formal_provenance_update.json。
原expected_tests.json仍封存原59个测试ID；full_expected_tests_v11.json冻结本版完整集合，
零跳过且集合完全一致才通过。原幂等保存测试按本票新合同改为同内容接纳，测试ID保留。
