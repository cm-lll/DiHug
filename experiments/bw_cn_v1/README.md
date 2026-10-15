# BW CN v1：候选边共同邻居聚合审查包

这是独立于原 v9 的开发实验包。目标是检验 learned common-neighbor feature aggregation 是否改善跨关系结构，尚无训练或生成质量结论。

## 内容与审查顺序

- `package/`：已交付 ZIP 的逐字节展开副本，包含冻结 IMDB 输入、完整源码、测试、配置、manifest 和中文说明。
- `package/v9_to_cn_v1.diff`：相对已验收 v9 的最小源码差异，建议先审阅。
- `package/hsgd/common_neighbor.py`：真实冻结 Xt 上的共同邻居提取、共享第一层上下文和聚合。
- `package/hsgd/model.py`、`computational.py`：模型与训练／采样接入。
- `package/hsgd/cn_entry.py`：检查、两臂从头训练、恢复和手动采样入口。
- `acceptance/`：实际运行的 CPU 测试、资源和补丁核验记录。
- `release/dihug_sparse_bw_cn_v1.zip`：原交付 ZIP；没有重新构建或更改内容。
- `review_provenance.json`：原始源码提交、发布哈希及本审查快照各文件哈希。

## 唯一方法变化

对真实 Xt 中 u-w-v 的共同邻居，使用现有 Transformer 第一层在仅含 Xt 真实边的图上得到 h_w；与中间节点类型、两条关系的对称编码、候选关系和时间一起编码，按候选边求和；附加中间类型计数，用无偏置读出残差加到原有边 logit。读出零初始化。无节点 ID、干净目标邻接、强制闭合或补边。共享第一层增加计算与梯度来源，但没有增加另一套 Transformer。

新增分支不依赖查询分块；原主干仍含查询边，不能宣称整个预测对分块不变。索引按全部真实楔建立，不截断共同邻居；稠密 Xt 的成本需要 BW 实测。

## 不变项与待验证范围

两臂均保留 8 层／8 头、隐藏宽 256/64/128、FF 256/64/256、AdamW lr=2e-4、weight_decay=1e-12、AMSGrad=False、microbatch=1、无梯度累积、FP32 网络／float64 概率核、T=1000 和 20 块查询。原损失、先验、精确 0/1 支持和原子 full_sweep 不变。默认两臂分别从头最多 10000 有效更新，不设八小时时限；不以该上限证明收敛。旧 v9 的 12365 更新权重不是同预算主对照。

baseline 参数 6527311；CN 参数 6537427；新增 10116（约 0.155%）。IMDB 固定输入为 1653 节点／4079 物理边，数据版本及哈希见 manifest／provenance。包含该冻结参考输入，不包含训练权重、生成结果或其他任务数据。

本地部署副本 41 项 CPU 测试通过（131.76 秒），含小图数值、梯度、入口、保存恢复及工程短链；补丁应用后 44 个 hsgd 文件逐字节核对。BW 检查、完整 IMDB 模型吞吐／显存、正式训练、T1000 科学生成和质量比较均未执行。原训练门针对 CN 的 old/new 调用采用同一 canonical upstream 实现，因此不能把它单独当作两个独立构图器的 CN 等价证明。普通模式恢复仅验证载入状态，不承诺未来随机数值轨迹逐位一致。

## 手动 BW 使用

先从 `release/` 下载 ZIP，并核对：

```
c045034ba970d6b7e211ac73ef9c1f9c7a0f6532e9cea919020277916df0fa37
```

在 notebook 终端解压后执行：

```bash
cd /root/private_data/gnn_projects
unzip -n dihug_sparse_bw_cn_v1.zip
cd dihug_sparse_bw_cn_v1
sha256sum -c SHA256SUMS
bash run.sh check --output /root/private_data/gnn_projects/bw_cn_v1/check01
```

完整检查／训练／进度／安全停止／恢复／手动采样命令见 `package/说明.txt`。检查通过后由用户手动启动两臂，单卡顺序运行；上传 GitHub 不启动任何训练或采样。不安装或替换 BW 厂商 Torch，不增加编译扩展。

源码来源提交：`080c124eb54dedf81001eb37589755c863e19ec8`。此审查分支基于远端 main 单独添加本目录，没有把整个历史实验分支带入，也不修改旧版本或中央状态。
