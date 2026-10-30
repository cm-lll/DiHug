HGEN全192节点参考图基线：先读REPORT_ZH.md和METRIC_DIRECTORY.md。科学源码source/逐字节绑定bfbd7b4；结果与CPU派生脚本单列，不改变冻结科学实现。

8个GAN拟合/16原图/16LCC已完成。原图run/*/sample_*/graph.npz；其edges为2×E、relations为1-based，data/*/reference.npz的edges为E×2。raw_adjacency和全部组装动作保留。完整指标evaluation.json与tables/。HGEN接收总边预算，四GAN seed共享每图一次Hin2Vec，原生梯度限制与零支持兼容见报告；不是原论文40%留出原样复现。

原科学运行固定本地路径，不要直接启动source/launch.py重跑。可独立复算CPU评价：OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B recompute_cpu.py --root . --output NEW_CPU_OUTPUT。使用现有numpy/scipy/networkx，不加载模型。模型/优化器.pt文件未打包；本地路径及哈希见local_weights_index.json。

SHA256清单见payload_manifest.json；ZIP外部哈希另附。没有凭据或无关项目文件。原始记录不改，未自动追加候选或样本。
