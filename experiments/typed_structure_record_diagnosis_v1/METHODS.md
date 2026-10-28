# 数据合同、字段缺失与复算

## 只读来源

三轮固定结果提交分别为b5b697361e389cd77d2fb09a84e77c0b6915407d、143e6bf80a9858ec21e512da2f1a03e02e3a5a38、93b41d954fcc46fd3732eb5c2780c76db0568f29。复用其已完成的模型/数据/配对审计，不重新加载权重、执行模型或全量审计。原本地运行目录逐项登记于`cell_sources.json`和CSV的`source_file`。

只核验本次直接使用的参考NPZ、validation/steps记录和保存状态的既有哈希回执。状态读取使用现有CPU Torch环境`torch.load(map_location='cpu')`，拒绝含model/optimizer的载荷；不导入项目模型、调度器或worker。引用源码仅作为文本复制，未执行。归档`evidence/*/saved_states.npz`是旧uint8状态的等值复制，不是重采样或新图；其`.sources.json`保留原pt路径、文件SHA256和实际state_t。无权重或凭据加入本包。

## 固定验证

24模型 × 3检查点(0/500/1000) × 3噪声时刻(1/500/1000)=216条总记录，每条10个schema关系，共2160条关系记录。每个时刻只有一份固定普通噪声输入；不是多次独立噪声平均。

- `ce_sum_over_S`来自稳定两状态logits CE总和除以完整schema S=18336。每份验证是单输入，不再乘normal2的1/2，也不除以活跃槽数。训练仍是两份sumCE/(2S)累计；本轮未更改任何目标。
- `positive_ce`、`negative_ce`是该关系被查询正/负标签位置的平均CE；样本数保留。总正/负CE均值按样本数加权，不能各关系简单平均。由两类平均CE乘数量复算总CE仅有FP32求和误差，最大绝对差见`analysis_receipt.json`，不替换原CE值。
- `expected_edges`是p0有边概率之和，非q或实际采样边数。表统一改名`p0_sum`。`true_edges`是本参考干净图该族数量。IMDB五族固定零支持未查询，正负样本数0，CE保留null，概率和0；不能把null写成完美分类。
- 记录未保存逐槽p0、logits、q、普通验证Xt数组/实际边数或独立查询摘要，不能逆推。q列全部空缺，JSON中用null、CSV用空单元格；不计算新后验。与无数据的0明确区分。
- 保存的`input_sha256`覆盖t及ids/query/f/edges/edge_families/labels。每数据集、同训练seed、同t下，六组×三个检查点的18条摘要一致；两个seed的完整摘要不同。摘要包含查询顺序，不能仅凭不同摘要断言Xt边集合本身不同；无数组就不重建。不同训练seed仍分开呈现，不能视为纯初始化干预。

## 生成状态

48条旧链，每条1000行日志，字段为time、remaining_time、backend、blocks、covered_slots、exact_coverage、step_seconds、timings、timing_scope；实际逐链清单见`field_inventory.json`。日志没有p0/q、总边数、关系边数或预测出生/消失。计时字段不是结构指标。

源码先完成反向t步，再`state=candidate; next_t=t-1`，随后写日志与检查点。因此`complete_t0499.pt`含X499，是t=500提交后的二值状态，可作为未来t=499的输入，不是t=500的输入Xt。反向完成步数=1000-state_t。

实际保存时间仅999/899/799/699/599/499/399/299/199/99/0，总计528状态。初始X1000没有保存，本轮不从seed重建。间隔内状态缺失，不进行采样重放或插值重建。曲线细虚线连接两个单链的观测点，粗虚线连接两链均值，仅辅助阅读；中间虚线不代表已观测轨迹。

状态是完整18336槽uint8向量，采用冻结TypedSlots的family-major顺序。按schema类型对计算每族容量并切片计数，同时独立构建端点映射，将每条链X0和已有graph.npz的边集合核对。参考NPZ为Ex2、relations从1开始；生成graph.npz为2xE、families从0开始，显式区分。原始node IDs只用于槽索引，不用于推断新特征。

全族及总实际边数、参考差值分别保存。`saved_interval_changes.csv`仅是相邻已存状态的边数差，不能称每步预计净增或累计出生/消失。参考数量只作对照，不要求每张带噪图或自由生成图逐步严格匹配，也不把参考外的边判成错误。

## 文件与复算

- `validation_total.csv`、`validation_relations.csv`：全部检查点、噪声时刻、类型关系和正负CE。
- `trajectory_total.csv`、`trajectory_relations.csv`：528个保存状态及5280个关系计数，含原路径/哈希。
- `validation_vs_generation.csv`：24模型最终验证与已有末图并列，不拼成因果轨迹。
- `TABLES.md`和`plots/`：完整24格简表、总边数轨迹及PubMed O/D逐关系曲线、IMDB MG固定验证。
- `input_consistency.json`、`field_inventory.json`、`crosscheck.json`：输入可比性、缺项和针对性核对。

现有环境Python3.9/Torch2.1 CPU读取；依赖实际版本另见`environment.json`，不安装升级。原始本地抽取命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/data2/lyh/miniconda3/envs/sparse_block/bin/python analyze.py \
  --project /data2/lyh/gnn_project/DiHug --output /tmp/records_new_output
```

公开包无需Torch/权重即可用现有NumPy复核表格和状态计数（`--bundle`指本包根目录）：

```bash
OPENBLAS_NUM_THREADS=1 python check_records.py --bundle .
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 python render.py --output .
```

后两命令只写本分析目录的交叉核对回执/派生表图；原实验日志不修改。首次抽取对参考NPZ字段约定有两次立即失败（误用families键、1基关系断言），修正后完整抽取成功；失败回执保留，不含科学调用。没有把已有全量验收当作本次新执行。
