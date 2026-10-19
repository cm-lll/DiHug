# IMDB／PubMed 联合 CN：双卡独立训练与轮次对照

本包基于 dd53b6431303263e740abedf67e01d7dfc13d15b 的 joint CN v1.1。
`hsgd/`、原三份核心测试及其兼容性测试依赖 `role_degree/` 均逐字保留；
生产入口 `dual/` 不导入角色引导。没有 M/K、硬解码、补连、度修复或闭合修复。
仅已做本地 CPU 小图工程验收，未运行 BW、真实图网络前向、正式训练或完整科学生成链。

## 配置与来源边界（启动前必读）

两数据集都是 CN，模型参数量均 6,637,105。8 层、8 头，节点／边／全局宽
256／64／128，FF 为 256／64／256，联合头 128，CN 内容 32。
输入输出 MLP 的**全部实际维度**见 `data/verified_inventory.json` 的
`dimensions.linear_layers`；这来自本地实际实例化，不是从配置名猜测。
无新增节点 ID、谱、环或位置特征。配置 dropout=0.1，但当前 vendor forward
没有生效的 dropout 操作；没有为了本次实验补开 dropout。

稳定 logits CE，损失 `L=(R/S) sum_{e in queried block} CE(z_e,y_e)`，R=20，
S 为完整 schema 槽数，精确固定支持合法项损失为 0；不平衡正负权重不变。
每次一个独立噪声时刻、一个查询块，microbatch=1、累积=1。
T=1000，余弦 offset=0.008，20 块分族均匀查询、冻结整步 Xt、统一提交。
FP32 网络／CE、float64 概率核，TF32 关闭，线程 1；正式训练和验证普通模式。

优化器：AdamW，lr=2e-4、weight_decay=1e-12、betas=(0.9,0.999)、eps=1e-8、
amsgrad=False；无调度器、无 LR warmup、无梯度裁剪。
这套优化器能追到固定基线的 **fusion_preflight 工程入口**。
**本地没有找到旧 6000 步正式 launcher、resolved_config 及优化器／种子回执**，
故不能声称已独立验证与旧正式实验严格同配置。文件中明确记为未核对；
启动命令的 `--accept-config-provenance-gap` 是对这一已披露边界的显式确认，
不是恢复旧权重或更改方法的开关。旧实验实际种子尚未认证，新 IMDB seed
2026101901 与本地已知历史 seed 2026101407／1603 不同。

IMDB/PubMed 训练 seed 分别 2026101901/2026101902；验证 seed
2026101911/2026101912；采样 seed 2026101921/2026101922。
每个数据集五个检查点共用自己的一个采样 seed。

## 数据（两份完整现有二值无向版本）

| 数据 | 节点 | 物理边 | 类型库存 | 全部槽 | 活跃槽 | 确定性空槽 |
|---|---:|---:|---|---:|---:|---:|
| IMDB | 1653 | 4079 | Director410 / Actor494 / Movie726 / Genre23 | 1365378 | 997313 | 368065 |
| PubMed | 1565 | 12284 | Gene313 / Disease643 / Chemical564 / Species45 | 1223830 | 1223830 | 0 |

均为 4 类型的 10 个无序类型对，每对至多一种关系，NPZ 关系编号 1..10，
模型内部 0..9。`schema` 明确保存类型对，逐关系密度由实际物理边数／槽容量推导。
active 仍由参考边数>0 决定，不冒称 schema 永久禁止该关系。
每轮查询 IMDB 约49865--49866，PubMed 61190--61191，全部活跃位置每步一次。

来源为已有 HGEN 完整具名预处理版本，未重新裁图或改二值化。
两参考均无自环、重边、孤立点，均单分量。IMDB 原加权／有向存储版本的
历史收据有 11 个自环；本包直接复用已封存的去环、无向物理边版本，不称原加权图。
`historical_inventory.json` 是旧来源记录；`verified_inventory.json` 另列本次实际
载荷及可读源文件重哈希，不能把历史收据当作本次重做预处理。
所有划分仍是同一参考索引 0，验证是同参考去噪，不是未见图泛化。

CN reference wedges：IMDB217005，PubMed1164378。独立前向噪声下每个 t 的
解析期望见 verified_inventory。反向状态不受这条期望曲线保证；原 max_wedges
及 max_query_paths 均2000000，超限停止，不截断，不自动改配置。

## 先检查，再由另一侧决定真实启动

解压后进入 `dihug_joint_cn_dual_rounds_v1`。默认 Python
`/usr/local/bin/python3.11`，保留 BW 厂商 Torch、已有 PyG，不安装或升级依赖。
所有路径示例是**新实验路径**；不要使用旧 v9／CN v2 输出目录。

```bash
export PYTHON=/usr/local/bin/python3.11
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bash run.sh inventory --dataset IMDB > /root/private_data/imdb_inventory_new.json
bash run.sh inventory --dataset PubMed > /root/private_data/pubmed_inventory_new.json

# 两项短测依次执行，每项含进程启动最多570秒，清理通常额外不超过3秒；
# 这些索引是平台当前可见范围内的本地索引，绝不解除设备隔离。
bash run.sh short-test --dataset PubMed --device-index 0 \
  --output /root/private_data/joint_cn_rounds_preflight/PubMed_check01 --max-seconds 570
bash run.sh short-test --dataset IMDB --device-index 1 \
  --output /root/private_data/joint_cn_rounds_preflight/IMDB_check01 --max-seconds 570
```

短测只有6次丢弃的工程更新（前三次预热、后三次稳定），一个完整20块反向步和
一个完整五时点验证面板；不是完整生成链、不保存为正式权重。输出区分初始化、
整理／噪声、前向、反向、优化器、后验及验证成本，设备计时同步。
`costs.json.estimates` 分开外推12000更新、13次验证、5条T1000链；
仅一个先验状态步，含 trace 诊断开销，不能保证密度变化后全链成本。三次稳定测量统计很有限。
若570秒内未完成返回非零、不提供伪造的成本估计。无本地 BW 实测，所以没有可靠小时数承诺。

## 审查通过后的双卡启动命令（本次交付不会执行）

```bash
nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /usr/local/bin/python3.11 -u launch_pair.py \
  --pubmed-device 0 --imdb-device 1 \
  --root /root/private_data/joint_cn_rounds_v1_run01 \
  --accept-config-provenance-gap \
  > /root/private_data/joint_cn_rounds_v1_pair01.log 2>&1 < /dev/null &
```

需要同一容器实际获配两张可见卡。启动器核对两个可见索引及可取得的UUID并打印；
保留 CUDA/HIP/ROCR 可见性变量，不申请或暴露额外卡。同一输出父目录的设备锁
阻止误把两项 pipeline 放同一卡。不要在同一卡另启 sample/训练任务；不要换一个
父目录绕过锁。若两个独立 notebook 各获配一卡，分别用下面单任务入口，各自索引0：

```bash
bash run.sh pipeline --dataset PubMed --device-index 0 --fresh-start \
  --output /root/private_data/joint_cn_rounds_v1_run01/PubMed_seed2026101902 \
  --accept-config-provenance-gap
# 另一获配设备／容器；同容器第二张卡时改为 --device-index 1
bash run.sh pipeline --dataset IMDB --device-index 0 --fresh-start \
  --output /root/private_data/joint_cn_rounds_v1_run01/IMDB_seed2026101901 \
  --accept-config-provenance-gap
```

首次必须 `--fresh-start` 且结果目录和 `.gate` 目录不存在。父入口先在独立进程做
严格确定性真实输入两次更新及短保存恢复检查（最多570秒），成功后在父进程普通
模式重新初始化正式模型，工程权重不导入、工程更新不计数。此门只检查短恢复，
不是独立的旧／新 CN 实现证明：CN legacy 本身也是 upstream 别名。

正式训练严格到12000次成功优化器更新；没有旧8小时限制。保存0步及每1000步
不可覆盖的检查点和哈希，验证在0、1000、…12000步；t固定为1/250/500/750/1000，
各时点完整覆盖20块。同一固定噪声面板及其绑定可重建核对；验证恢复全部训练RNG。
记录全S分母CE、逐关系正负CE及数量、常数先验CE、概率校准、完整预计边数及
分类型期望度尾部。单次随机t/query的训练loss不是验证loss。

训练完成后**每卡顺序**进行u3000/6000/9000/10000/12000的完整T1000 raw采样。
每点仅一张；不宣称多样性、稳定性、收敛或跨seed鲁棒性。每步20块冻结同一Xt，
全部提交后原子保存状态与RNG；普通模式恢复允许后续浮点轨迹差异。

## 进度、安全停止、恢复及单独采样

```bash
tail -n 5 /root/private_data/joint_cn_rounds_v1_run01/IMDB.log
bash run.sh progress --dataset IMDB \
  --output /root/private_data/joint_cn_rounds_v1_run01/IMDB_seed2026101901
bash run.sh stop --dataset IMDB \
  --output /root/private_data/joint_cn_rounds_v1_run01/IMDB_seed2026101901
# 采样期间要停止当前采样目录（不重跑已提交反向步）
touch /root/private_data/joint_cn_rounds_v1_run01/IMDB_seed2026101901/samples/u003000/STOP
```

等待日志显示 stopped、安全检查点已写出及进程退出。STOP只作用于指定新任务；
SIGINT／TERM／HUP同样请求安全退出，不保证能捕获SIGKILL或机器断电。
先确认原进程已退出，明确删除**本次**STOP，然后恢复：

```bash
rm /root/private_data/joint_cn_rounds_v1_run01/IMDB_seed2026101901/STOP
bash run.sh pipeline --dataset IMDB --device-index 1 --resume \
  --output /root/private_data/joint_cn_rounds_v1_run01/IMDB_seed2026101901 \
  --accept-config-provenance-gap

# 只训练、不自动采样：首次用 train --fresh-start，续训用 resume。
bash run.sh sample-checkpoint --dataset IMDB --device-index 1 \
  --run /root/private_data/joint_cn_rounds_v1_run01/IMDB_seed2026101901 \
  --update 6000 --output /root/private_data/joint_cn_rounds_manual_samples/IMDB_u6000
# 续采样同一目录，另加 --resume；不要更换seed或擅自换检查点。
```

完成训练的 pipeline 恢复不会重训，继续尚未完成的预定采样。状态/图/评价不全不能
以 completed 字段跳过。异常、OOM、非有限值、来源改变均非零退出并保留有效文件；
失败更新不计数、不保存可能部分更新的模型，最近有效持久检查点仍可供审查。

评价只读取raw图，输出各关系边数、逐类型两端度直方图与未缩放W1、完整联合
关系度经验频数及TV、度尾部、分量/LCC/孤立点、类型三角形、MP1/2/3计数和
模式TV。MP是节点不重复的简单路径，完整反向合并，不排序类型，逐图归一化；
长度3用类别组合乘积再减三角回返，已用独立DFS小图核对。确切运算预算2000万，
超限停止，不抽样近似。IMDB另列D-M-A路径闭合及去重对闭合（分子分母均保存）；
PubMed不套用IMDB名称。没有新增质量阈值或综合分数。

## 本地复验

```bash
/usr/local/bin/python3.11 test_release.py --output /root/private_data/joint_cn_cpu_test01
```

只用小图、短日程测试；冻结完整测试ID、清除继承的pytest筛选设置，零跳过才通过。
不安装依赖，不把CPU小图结果当作BW真实吞吐或生成质量证据。
