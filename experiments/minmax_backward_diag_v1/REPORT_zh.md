# 首层节点min/max反向诊断：输入前置条件未满足

本票2026-10-28 00:36:50 Asia/Shanghai开始，固定截止00:51:50。实际结束时间和耗时见completion.json。
在指定初始权重核验阶段停止，未执行模型前向、反向、GPU算子或20次重复。

## 阻塞证据

v2的probe.py第96行把初始state_dict放入内存变量initial，第114行从该变量重载；只在第139行保存after1.pt。
v2全部六个落盘checkpoint均为after1.pt，发行包的local_engineering_checkpoints.json也只登记这六个文件。
IMDB_A/after1.pt实际读取确认updates=1，keys为model/optimizer/rng/updates/binding/purpose，不包含初始权重副本。
该文件SHA256为1921fa345050a33697e54224eb7d9b515730e9cf3c0cb65b24abe6768f3c328f。
本票指定的“v2已有IMDB A初始权重”没有被上一票持久化。这是上一票证据保留遗漏，不能用同seed重新初始化、v1初始文件或after1权重冒充。

统一输入NPZ已找到，SHA256为9d249b717066ec78981f3e01e33405766fd0f85cbcd686c326ca3bf7b9ff4635。
第一份输入包含13120个唯一槽、313条Xt真实边、端点/关系/标签；时间500由原冻结probe代码及回执绑定，NPZ本身未存标量t，不能伪称原数组包含该字段。

## 已完成的静态核查

模块为首层backbone.tf_layers.0.self_attn.x_y，即SparseXtoy。
节点min通过mi=-global_max_pool(-X,batch)实现；max为ma=global_max_pool(X,batch)。
global_max_pool将二维输入按节点维度dim=0聚合，调用PyG scatter(reduce='max')；size默认由batch最大值计算。
已安装PyG2.3.1的分派代码：有torch-scatter且输入为CUDA并需要梯度时，调用torch_scatter.scatter；CPU或无需梯度则调用原生scatter_reduce_。
torch_scatter.scatter_max的Python包装调用torch.ops.torch_scatter.scatter_max，能返回值和argmax索引，PyG层仅保留值。
实际环境：Torch2.1.0+cu121，PyG2.3.1，torch-scatter2.1.2+pt21cu121。
模块路径、Python源码、扩展so哈希均保存environment_and_dispatch.json。
这是已安装实现与冻结代码的静态调用核查，不是本轮真实GPU调用跟踪，也不是非确定性已证实。

min捕获时应明确内部U=-X、V=max(U)、mi=-V；上游梯度g_min对应内部g_V=-g_min，映射回X还要再取负。
本轮没有捕获张量，未实际验证这组符号或梯度守恒。

## 未完成项与结论边界

未执行：真实聚合输入/上游梯度捕获、GPU20次重复、极值索引变化、唯一/并列分层、梯度守恒、CPU数值比较、边级聚合。
不能称“算子稳定”，也不能称“已证实并列极值梯度分配导致原异常”。旧v2失败结论原样保留。
没有提出或实施新的梯度规则，平均分配仅保留为此前条件性建议。

optimizer.step=0，正式训练=0，生成图=0，GPU初始化/分配=0，无本票GPU资源需要释放；CPU检查进程已退出。
继续所需条件：找到可认证的原初始权重；若确实未保存，需要用户另行明确是否允许按冻结过程重建并作为重建实验记录，不能在本票自行替换。
本票不重新计时，不等待补文件而自动启动后续计算。
