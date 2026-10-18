# 独立多尺度联合CF＋解析尾部引导 v1（待审查）

唯一源码起点：acbbddb5cdbcbaa2ae129bcd59e434e6a298b130/bw_stepwise_role_v1。
未使用另一位助手的实现、ZIP或其新方案源码。六状态“旧代理下降但Genre独立64与W1
变差、增大强度未解决”是用户提供的已有诊断，本轮没有重新实测。
保留模型、稳定logits CE、噪声、Bayes后验、T1000、20块计划和冻结Xt原子提交。
新增可训练参数0。A/C用原生无CN权重，B/D用原生CN权重；不加载或重训BW私有权重。

## 冻结目标，不按六状态效果选参数

仅有按真实类型c分组的完整关系度向量d及频数rho_c(d)，不含节点分配、ID或真实边身份。
目标来自已允许训练参考，称“已知目标分布的引导生成”。完整类型对槽才支持；
节点特定掩码继续拒绝。同类型容量为n_c-1。固定pi=0/1不变。

### 多尺度联合特征函数

每类型/关系尺度 sigma_cr=1+sqrt(E_rho[d_r^2])。
32基础方向：先每个非零容量、pi>0入射关系的单位轴；余下方向采用
PCG64 SeedSequence([2026101801,c])，各入射坐标独立sign×Uniform(.25,1)/sqrt(k)。
因此k>=2时联合方向同时涉及多个关系；k=1时也保留不同幅度，不退化为复制同一频率。
使用尺度倍数{.5,1,2}，omega=倍数×方向/sigma，合计96个频率。
只读训练统计，不读六状态、独立64评价频率或指标。C/D与所有时刻相同。

s=t-1；a_s为原累计保留系数，b_r=a_s+(1-a_s)pi_r，z_r=(1-a_s)pi_r。
Phi*_ck(s)=sum_d rho_c(d) product_r
  [1+b_r(exp(i omega_ckr)-1)]^d_r
  [1+z_r(exp(i omega_ckr)-1)]^(N_cr-d_r)。
先在整条联合角色向量内相乘，再求角色平均。
Phi_ck(q)=n_c^-1 sum_v product_(e incident v)[1+q_e(exp(i omega_ck,r(e))-1)]。
L_CF=mean_(c,k)|Phi_ck(q)-Phi*_ck(s)|^2/4，故归一到[0,1]。
Phi为期望经验CF，不是CF(期望度)。一条无向边同时影响两端。

### 解析平滑尾部

每类型c建立各非零入射关系度渠道，以及总度渠道。只有一个关系时总度完全重复，
只算一次；零容量/pi=0恒零渠道排除。不同渠道是关系坐标子集，不引入逐节点目标。
按匿名参考计算该渠道D的加权下分位D90=min{x:P(D<=x)>=.9}。
冻结p_ch=min(.5,3/max(1,D90))。此p是尾部探针，不是改动边概率的epsilon。
h_p(D)=Pr[Binomial(D,p)>=3]。

给定独立Bernoulli(q)边，E[h_p(D_v)] = Pr[PoissonBinomial({p*q_e:e incident v})>=3]。
目标mu*_ch(s)=sum_d rho_c(d) Pr[sum_(r in ch)
  Binomial(d_r,p*b_r)+Binomial(N_cr-d_r,p*z_r)>=3]。
mu_ch(q)=该类型全部节点对应尾部概率的平均。
L_tail=mean_c mean_(去重渠道ch) (mu_ch(q)-mu*_ch(s))^2，范围[0,1]。
最终L=.5*L_CF+.5*L_tail；0.5/0.5、三个尺度和p规则都是待验证工程选择，不是最优性结论。

尾部采用概率状态[0,1,2,>=3]的非负卷积。相同概率的二项目标用幂平方；
生成侧用分组前缀/后缀扫描。导数：
  d Pr(PB(pq)>=3)/dq_e = p*Pr(PB(其它p*q)>=2且<=2)。
即其余边恰好2条，由前缀0/1/2与后缀2/1/0卷积求得。
无除法、无1-CDF抵消，无截断真实尾部、无期望度替代。每个统计仅当前完整q参与。

## 原采样位置与控制器保持

原20块固定Xt收齐p0，释放网络图→原后验q→新L及梯度→有界调整→原后验→原uniform采样。
对收集p0的Bernoulli logit求梯度：dL/dq * (b1-b0) * p0(1-p0)，
b0/b1仍用原kernel.reverse计算并检查仿射关系。t1原后验q=p0。
没有网络反传，没有额外采样RNG；关闭引导直接委托原流程。

原控制器不改：delta_t=.1*a_s；无穷范数归一化；最多8候选/.5回溯；
下降须>1e-12+1e-9*|L|；梯度<=1e-14或线性化下降不可分辨则跳过。
非有限数值中止/回滚；零梯度反例保留。数值饱和p0=0/1不解锁，另记槽数。
logit坐标为logit(已收集p0)，避免重新计算softmax带来未记账的基点变化。
每步KL是条件raw独立转移的KL，不能当成整链KL保证。

频率|omega|<=2<pi，单因子模长>=cos(1)>0；先核对该界再除。
整体乘积用log积，绝不除以整项乘积。目标/统计下溢分别记账。
尾部只做非负运算，p<=.5保证单失败因子>=.5，但长乘积仍可能下溢。
出现统计下溢按原控制器保守跳过，不把残余数值信号归一化放大。
DP质量误差>5e-11停止，不裁剪/重归一化概率。

CF时间O(96S)，工作空间O(S+N)；尾部扫描O(S log最大度)，空间O(S)，
按类型/渠道依次处理，不构造N×N×K张量。目标按s惰性缓存；首步计算计入成本。
CPU参考实现，无GPU加速。GPU数值及控制器决定一致性未测，不宣称支持。

## 独立评价与预先判定

旧Genre独立64频率必须由原始冻结工件提供；同时提供原始工件哈希、频率数组、
归一化与定义。缺失不自行生成64频率。若原W1定义不同，入口拒绝静默替换。
本实现的精确W1：先求每节点完整Poisson-binomial度PMF，再对类型内节点平均成
期望经验度分布，与前向目标PMF比较 sum_k |CDF_pred(k)-CDF_target(k)|。
它是“期望分布的W1”，不是“生成图W1的期望”，也不是“每节点期望度分布W1”。
全系数FP64递推，无MC/FFT/截尾/renormalize。O(N*容量^2)可能是主要成本，
每列检查时间；超时只给未完成，不能改成近似值。除Genre外也列所有去重渠道W1。

六状态固定A/B×t500/20/1，不替换、不挑选。Genre64差异的数值判别为
2e-10+1e-9*|before|；W1为1e-8+1e-9*|before|加前后两侧各自
64*eps*capacity*(capacity+nodes+role_rows+1)的保守运算量舍入裕度（非区间证书），
只用于区分浮点不可分辨变化，
不是新质量合格阈值或统计显著性。
六状态全部完成且两个独立指标均无可分辨退化、至少一个状态两者同时改善，
才描述为“这个固定面板显示一致改善”；任一退化必须说存在取舍；无可分辨改善则
证据不足。输入/工程/超时只报协议失败或未完成，不能用通过的子集宣布成功。
代理下降不进入上述独立收益判据。完整记录关系边数期望、梯度、KL、步长及成本。

## 900秒边界与命令

没有旧绝对截止时间。每次独立调用最多900秒，含进程启动、数据核验、目标准备、
后验、统计梯度、回溯、评价和保存；工作进程提前8秒进入软截止，父进程保留清理时间。
不提供训练或完整科学链入口。输出必须新目录，所有输入结束时重新哈希。
安全停止：在新输出目录创建STOP文件，或前台Ctrl+C；只保留完整计算记录，
不重发、不从部分面板自动续跑。若硬杀，controller非零/未完成优先于任何部分结果。

```bash
bash multiscale.sh test --output /root/private_data/gnn_projects/multiscale_role/test01 --seconds 120

# 只有原始六状态和独立评价工件齐全、审查通过后才手动使用：
bash multiscale.sh short \
  --manifest /root/private_data/gnn_projects/SAVED_SIX/six_states.json \
  --manifest-sha256 REPLACE_WITH_VERIFIED_MANIFEST_SHA256 \
  --evaluation /root/private_data/gnn_projects/SAVED_SIX/original_evaluation.json \
  --evaluation-sha256 REPLACE_WITH_VERIFIED_EVALUATION_SHA256 \
  --output /root/private_data/gnn_projects/multiscale_role/short01 --seconds 900

touch /root/private_data/gnn_projects/multiscale_role/short01/STOP
```

默认Python /usr/local/bin/python3.11，OMP/OPENBLAS线程均1；不安装任何依赖。
需要的具体输入格式见INPUT_CONTRACT.md。路径是待填示例，不代表本地已存在六状态。
本轮仅实现/CPU测试/打包上传，未连接BW、运行六状态科学短测或完整链。
