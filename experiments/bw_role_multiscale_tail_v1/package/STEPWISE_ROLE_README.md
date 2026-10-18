# 逐步匿名联合角色度引导 v1

基于固定 joint CN v1.1 `dd53b6431303263e740abedf67e01d7dfc13d15b`。
这是独立采样适配器；hsgd、原 role_degree、原39项测试均逐字节保留。
旧 role_degree/role_sampling.py 仍是历史 t=1 版本，不是本轮入口。
本轮入口为 stepwise_role/sampling.py，所有 t 都调用引导。
不训练、不运行正式完整链、不读取 BW 私有权重，不联网、不安装依赖。
旧 run.sh/build_release.py/说明等保留为历史来源；请使用新的 stepwise.sh。

## 定义

按类型 c 保存完整关系度向量 d 的压缩频数 rho_c(d)，不保存节点分配。
容量 N_cr 由完整类型对 schema 决定，同类型不含自环；缺失或节点特定合法槽
暂不支持，Layout 会明确拒绝，不套用统一容量。精确 prior=0/1 保持固定。
目标只来自允许的训练参考，属于已知目标分布的引导生成。

下一时刻 s=t-1，b_r=a_s+(1-a_s)pi_r，z_r=(1-a_s)pi_r：

Phi*_c,k(s) = sum_d rho_c(d) product_r
  (1+b_r(expm1(i omega_crk)))^d_r
  (1+z_r(expm1(i omega_crk)))^(N_cr-d_r).

先在每个完整角色向量内相乘，再按频数平均；不是分别匹配一维直方图。
复用旧冻结 K=32、seed=2026101603：按关系的尺度 1+RMS目标度，
先每个可动入射关系轴方向，再每类型独立 SeedSequence 的 uniform(-.5,.5)
联合方向，全部除以该尺度。C/D 和所有时刻相同。完整目标表预计算1001时刻。

给定当前完整 Xt，原后验 q_e(p0_e) 条件独立抽样下：
Phi_c,k(q) = (1/n_c) sum_(v:type c) product_(e incident v)
  [1+q_e expm1(i omega_crk)].
L(q,s) = mean_(c,k) |Phi_c,k(q)-Phi*_c,k(s)|^2。
一条无向边一个变量，同时影响两端。原后验对 p0 仿射，
q=b0+(b1-b0)p0，其中 b0/b1 用未修改的 kernel.reverse 在可行基点算出，
每次核对仿射重建误差<=5e-14。
梯度 g_e=dL/dq_e*(b1-b0)*p0_e*(1-p0_e)。

使用收齐的受支持 p0 的 Bernoulli logit 坐标 z=logit(p0)，
不是对神经网络激活反传；这也避免把 FP32 softmax 重新用 FP64算一遍造成
未记录的基点改变。数值 p0=0/1（包括非结构性饱和）保持不动并单列。
原始两 logits 共同平移与此坐标无关。

delta_t=.1*a_s；z' = z - delta_t*g/||g||inf；最多8个候选
delta_t/2^j, j=0..7；不是初试后另加8次。接受条件：
Lnew < Lold - (1e-12+1e-9*abs(Lold))。
||g||inf<=1e-14、线性化下降预测 delta_t*sum(g^2)/||g||inf 不超过
该数值门槛、无可动槽或出现乘积下溢，均不归一化放大，明确跳过。
所有候选失败原样返回。非有限数值标记 numerical_failure，采样适配器停止并回滚 RNG。
除频率受限的非零单因子外不作任何除法：
Re(1+q*expm1(i omega))>=cos(.5)>0，先检查界再除；
绝不除以可能下溢的整项乘积。不加 epsilon 或裁剪概率。

## 数据流与 API

1. 构造一次 Layout、匿名 profile、freeze(profile,layout,alpha) 和 StepwiseGuide。
2. model.eval()；每步 reverse_step(...,arm='C'或'D',guide=guide)：
   固定Xt和模型版本→原plan→原先抽全槽uniform→同一FrozenTypedContext内20块预测。
3. 收齐p0并核对活跃槽恰好一次，非活跃槽保留精确支持；关闭上下文，释放网络图。
4. 原后验→全图统计/梯度/有限回溯→原后验→原uniform采样→返回完整待提交状态。
5. 调用方仅在整步成功后原子保存/提交，再进入t-1。任何异常含中断均恢复本步前所有RNG。
   不能跨查询块缓存普通主干隐藏表示；原CN上下文缓存生命周期不变。

```python
from stepwise_role.core import Layout, freeze, StepwiseGuide
from role_degree.role_statistics import joint_profile
from stepwise_role.sampling import reverse_step, binding, save_checkpoint, load_checkpoint
layout = Layout.canonical(case['space'].types, case['space'].pairs, case['prior'])
families = case['space'].encode(case['clean'])[0]
profile = joint_profile(layout.types, case['clean'], families, layout.pairs)
frozen = freeze(profile, layout, kernel.alpha)
guide = StepwiseGuide(layout, frozen)
# model为原生加载的joint CN v1.1 A或B权重，不改变任何参数
model.eval()
next_state, record = reverse_step(model, case, state, kernel, t, 20, seed, rng,
                                  device, arm='D', guide=guide)
```

A/C共用原生无CN权重；B/D共用原生CN权重。A为匹配联合框架，不是旧v9。
关闭引导 A/B 直接委托原 reverse_step，连随机数消耗也不改。
引导不消耗额外RNG。角色引导对全部时刻生效，但早期微弱梯度可以安全跳过。
本地只有小图随机权重；6000更新两套BW最终权重尚未导入或验收。
validate_arm 对调用者传入的6000更新最终权重receipt只检查元数据和原生架构；
不是独立权重来源证书。真实checkpoint/source/config/data哈希须由外部已验收加载入口核对，
本包不根据自报receipt声称已读BW权重。

binding/save_checkpoint/load_checkpoint 绑定代码、模型状态、架构、数据、
Python/Torch/PyG/NumPy实际版本、模式、完整目标/频率/时间表、backend、seed和20块规则。
原子保存已提交状态、剩余时间、全部相关RNG；SHA覆盖包含RNG的序列化载荷。
普通模式恢复只保证加载的保存内容和合同一致，后续轨迹不保证逐位等同。
仅信任本地产生的Torch检查点（pickle），不可载入来源不明文件。

## 测试和成本

CPU入口，总上限480秒，失败非零退出，不会自动开始BW、正式训练或完整链：

```bash
bash stepwise.sh test --output /root/private_data/gnn_projects/stepwise_role_checks/check01 --max-seconds 480
bash stepwise.sh inventory --output /root/private_data/gnn_projects/stepwise_role_checks/inventory01 --max-seconds 120
```

test 仅小图工程网络调用，无优化器更新；含原39回归、前向目标枚举、后验梯度、
两种模型关闭等价、4步工程链、部分步中断和新进程续跑等。不同Torch/设备尚未实测。
inventory只读取完整IMDB、冻结匿名目标，绝不调用模型或生成图。

cost仅接受已有完整状态/对应p0的NPZ，绝不代为运行完整链补输入：
state和p0为全1365378槽同序数组；t为1..1000标量；alpha为原1001项日程；
data_sha256为参考文件SHA字符串；source_record为具体生成来源/权重/查询/状态绑定说明。
这些来源须独立核实，命令只核对用户确认哈希并记录来源文本，不把文本当数学来源证书。

```bash
bash stepwise.sh cost --input /root/private_data/gnn_projects/FIXED_SAVED_INPUT.npz \
  --expected-sha256 REPLACE_WITH_VERIFIED_INPUT_SHA256 \
  --output /root/private_data/gnn_projects/stepwise_role_checks/cost01 --max-seconds 480
```

这个CPU概率短测只测统计/梯度/回溯与进程峰RSS，不包含网络收集和设备显存。
没有合法匹配输入时不要执行cost，不要拿其他T/其他模型的q冒充p0。
网络收集使用 StepPredictor 的既有分项计时，sampling.reverse_step(trace=True)
可记录真实单步构图/前向/导出开销，但本轮未获得匹配IMDB状态，未运行该设备短测。
未预留真实长链的授权或入口，不自动搜索步长/频率/时间表。

## 科研边界与后继（未执行）

Phi*为前向总体目标，q条件于单张Xt；即使模型正确也不要求逐状态相等。
有限32频率不确定唯一分布，L(E[经验CF])不同于E[离散统计平方误差]。
代理下降、每步logit<=.1或每步raw条件KL均不保证整链扰动小或生成质量好。
这是新引导律，不是原分布或目标条件律的精确采样，也不是CID/MASK直接复现。
未增加末端修边、补连、边数/度硬限制或共同邻居截断。

下一阶段先在用户确认的固定真实状态做有界IMDB概率收集+引导成本检查；
再另行决定四组同种子同预算生成。共同评价离散联合角色分布、关系边数、度尾部、
LCC/连通和类型闭合，记录逐步规模；不能只看用于优化的32频率。
