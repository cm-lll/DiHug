# 动态匿名联合角色匹配 v1：仅实现与有界诊断

此包**不训练、不调用真实模型、不启动1000步链、不连接BW**。
旧121个文件保持原样；新入口是run_dynamic.py。不要用继承的旧入口
代替本版入口。CPU需要当前已有NumPy、SciPy；不自动安装任何依赖。
本地实测依赖及证据在evidence_dynamic，不能当成BW或真实六状态结果。

## 本地/已审核CPU环境：完整工程回归

```bash
python verify_dynamic.py --output /tmp/dihug_dynamic_tests_01
```

入口核验包文件、完整冻结用例集合和零跳过，并隔离继承PYTEST筛选环境。
结果目录必须新建；测试包含小图神经前向与4步工程链，不是科学样本。

## 待审查后：只读导出可移植真实输入（不会跑引导）

下面命令由有权限读取notebook文件的人手动执行；Sol本轮没有连接BW。
导出只哈希/复制数组和收据，不调用模型、训练或生成。输出必须换成
全新路径，且不能在旧输入目录里面。预计体积以实际NPZ为准；按常见
数组dtype估计六份解压载荷约400MB，ZIP实际大小取决于压缩。

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  /usr/local/bin/python3.11 run_dynamic.py pack \
  --export /root/private_data/gnn_projects/role_mt_v1_2_export_20261018T091055Z \
  --short-run /root/private_data/gnn_projects/role_mt_v1_2_short_20261018T091510Z \
  --output /root/private_data/gnn_projects/dynamic_role_portable_01 \
  --max-seconds 900
```

交付portable_inputs.zip及summary.json中的transfer_zip_sha256、
portable_sha256到Sol可读位置即可；不需要上传任何私有权重。
原manifest/metadata不变，私有权重仅保留历史哈希收据，明确未重新认证。

## 输入到达Sol并复审后：先核验，再B1，再冻结六状态

以下路径及SHA请填实际导出回执值，不可自算后当成原来源认证。
max-seconds900涵盖所有步骤，不是每项评价各900秒。每次新输出目录。
本轮未在真实输入执行以下命令；不应把它们当成已通过。

```bash
python run_dynamic.py check --portable /path/portable/portable.json \
  --portable-sha256 EXPORT_RECEIPT_SHA --output /tmp/dynamic_check_01 --max-seconds 900
python run_dynamic.py b1 --portable /path/portable/portable.json \
  --portable-sha256 EXPORT_RECEIPT_SHA --output /tmp/dynamic_B1_01 --max-seconds 900
python run_dynamic.py six --portable /path/portable/portable.json \
  --portable-sha256 EXPORT_RECEIPT_SHA \
  --b1-receipt /tmp/dynamic_B1_01/summary.json --b1-receipt-sha256 B1_RECEIPT_SHA \
  --output /tmp/dynamic_six_01 --max-seconds 900
```

check核对六状态但不做引导/完整PMF评价；B1开发检查之后six强制绑定
相同源码、目标和输入。没有搜索步长、频率或重新挑状态的入口。

```bash
touch /tmp/dynamic_six_01/STOP
```

安全停止不承诺完成所有指标，部分文件不算通过。每次独立诊断可在
固定输入上重做同一步，但没有续跑原完整链的入口。小图采样接口另经
完整提交、局部中断回滚和新进程恢复测试；普通GPU非确定性仍未测。

设计、归一化、梯度及并列限制见DYNAMIC_ROLE_DESIGN.md；关键限制是
不同角色等成本时具体梯度可能随存储顺序变化，且匹配期望度不等于
匹配离散图的联合度分布。代理下降不构成保真成功。
