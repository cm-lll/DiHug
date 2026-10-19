# DiHug current research status

2026-10-19 IMDB/PubMed joint CN双卡轮次包：本地与独立解包各59 CPU检查通过。
独立experiments/joint_cn_dual_rounds_v1，基线dd53b643；28个模型/损失/噪声/采样/兼容测试文件逐字保留。
完整IMDB1653/4079、PubMed1565/12284及类型/关系/文件哈希已核实，均6637105参数。
两套从头12000更新、0及千步永久检查点、3000/6000/9000/10000/12000各一张raw图。
全槽固定同参考去噪验证、独立RNG、顺序采样、原CN路径上限；无引导/修复/补连。
source/unpack59项零失败零跳过，17.941/18.351秒，峰RSS475.74/475.04MiB。
开发失败及修复保留；ZIP SHA85ed407048cfd9f493a92312ac6fa3ee63503add13158b0222d4b63cc340b511。
旧6000正式入口/实际resolved_config及seed回执缺失，优化器只认证至工程AdamW入口。
BW/真实网络前向/正式训练/完整科学链均未执行；GPU吞吐显存及全链成本待设备短测。
证据results/joint_cn_dual_rounds_v1_20261019；独立审查交付后停止，不自动正式启动。
