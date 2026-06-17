# 当前 t 下「理想可达全图」与损失形式

**与训练/采样的关系**：模型采用**局部训练**（每步只预测一块 query 边），采样时通过**多轮生成**拼出全图。此处的「全图」指**当前步的完整图** = 当前噪声图 + 本步 query 上的赋值；ideal 即在该全图上定义的**局部优化**：只调整 query 边，使该全图在**规定的全局结构指标**（三角形数量、聚类、度分布等）上最接近真实图 A。

---

## 1. 数学上求 ideal 的过程

**设定**：给定时间步 t，有  
- 真实图 A（clean）  
- 当前噪声图 G_t：边集与 A 相同，边类型为扩散噪声  
- 本步只对 **query 边** 做预测，其余边保持 G_t 的噪声  

**目标**：在「只改 query 边」的约束下，找一个 query 边的赋值，使得 **当前步全图** \(G(q)\) 的结构（三角形数等全局指标）与 A 最接近。

记：  
- \(\mathcal{Q}\) = query 边下标集  
- \(E_{\text{full}}\) = 全图边属性（\(|E| \times C\)，C 为边类型数）  
- 非 query 部分固定为 \(E_{\text{full}}^{(\neg \mathcal{Q})} = \text{noisy}\)，只优化 query 部分 \(q \in \mathbb{R}^{|\mathcal{Q}| \times C}\)（在单纯形上）

**理想问题（离散）**：


\[
q^* \in \arg\min_{q \in \{\text{one-hot}\}^{|\mathcal{Q}|}} 
L_{\text{struct}}\big( G(q), A \big)
\]



其中：  
- \(G(q)\) = 全图边为 (非 query = noisy, query = q)  
- \(L_{\text{struct}}\) = 结构损失（relation matrix / metapath2,3 / subtype degree 等与 A 的 L1 加权和）

**可微近似（实现中）**：  
- 把 \(q\) 松弛为概率 \(q \in \Delta^{|\mathcal{Q}| \times C}\)，用 \(G(q)\) 的 soft 统计算 \(L_{\text{struct}}(G(q), A)\)（可微）。  
- 从初始 \(q^{(0)}\)（如 query 处用真实 one-hot）出发，做 **一步梯度下降**：
  

\[
  q^{(1)} = \mathrm{proj}_{\Delta}\big( q^{(0)} - \eta \nabla_q L_{\text{struct}}(G(q^{(0)}), A) \big)
  \]


- 取离散：\(\hat{q}^* = \arg\max_c q^{(1)}\) 作为本步 **ideal** 的近似。  
- 用 \(\hat{q}^*\) 得到 **ideal 全图** \(G^*\)（非 query = noisy，query = \(\hat{q}^*\)）。

---

## 2. 损失：预测图与噪声图「到理想图」的比值

若直接用 \(L_{\text{struct}}(\text{pred}, \text{ideal})\)：  
- 当 \(L_{\text{struct}}(\text{noisy}, A)\) 很大时，\(L_{\text{struct}}(\text{pred}, \text{ideal})\) 的绝对数值也容易大；  
- 预测相对噪声的改善在梯度里可能不明显（真实图与噪声/预测的结构差都很大，改善被尺度压掉）。

因此改为 **比值损失**（目的：让模型学习**基于当前(noisy)尽可能靠向真实图**）：


\[
\mathcal{L}
= \frac{
  L_{\text{struct}}(\text{pred}, G^*)
}{
  L_{\text{struct}}(\text{noisy}, G^*) + \epsilon
}
\]



解释：  
- **分子**：预测图到理想图的结构距离  
- **分母**：当前噪声图到理想图的结构距离（+ ε 防除零）  

性质：  
- 若 pred 比 noisy 更接近 ideal，则 \(\mathcal{L} < 1\)  
- 若 pred 与 noisy 一样远，则 \(\mathcal{L} \approx 1\)  
- 最小化 \(\mathcal{L}\) 等价于让「预测相对当前噪声」更靠近 ideal（当前 t 下最接近真实 A 的可达目标），且尺度由分母归一，避免改善被掩盖。

实现：当前训练已改为仅用 query 上 CE(pred, ideal)，不再使用上述比值损失；ideal 由 `structure_only_global=True` 时在固定 query 边数下对结构损失做梯度下降得到。

---

## 3. 是“最小化”还是“最大化”？可微吗？损失 vs 目标函数

- **当前实现：最小化比值**  
  损失取为  
  

\[
  \mathcal{L} = \frac{L_{\text{struct}}(\text{pred}, G^*)}{L_{\text{struct}}(\text{noisy}, G^*) + \epsilon}
  \]

  
  训练时对 \(\mathcal{L}\) 做最小化。  
  - 比值越小 → pred 相对 noisy 越接近 ideal → 越好。  
  - 所以仍然是「损失越小越好」的常规逻辑。

- **可微性**  
  - 分子 \(L_{\text{struct}}(\text{pred}, G^*)\) 对 pred（进而对模型参数）可微。  
  - 分母对 pred 用 `.detach()`，当作常数。  
  - 故 \(\mathcal{L}\) 对模型参数可微，可以正常反向传播。

- **损失 vs 目标函数**  
  - **目标函数**：你希望达到的目标（例如“预测图越接近理想图越好”）。  
  - **损失函数**：训练时实际被最小化的量。约定俗成是「越小越好」。  
  - 若目标是“某个指标越大越好”，可以定义损失 = 其负值，然后最小化损失等价于最大化目标。  
  - 在这里，目标 =「比值小 / pred 接近 ideal」，损失 = 该比值，最小化损失即达成目标，无需再取负。

---

## 4. 固定 query 内「显式边」边数：fix_k（增删成对、边数中性）

**动机**：若 ideal 在 query 上可以任意加减边，训练目标会经常是「在噪声图上加边」以靠近真实图，模型很少学到「去边」；采样时 merge 会偏向边数膨胀。因此约束 **ideal 在 query 上的显式边（存在边）数 = 噪声图在 query 上的显式边数**，即只做「在 query 内的重分配」（删一条、补一条），**边数中性**，与采样时的加删对称更一致。

**目标**：控制边数使**多轮时不同轮次的场景边数一致**——即每轮面对的「当前图」的显式边总数保持稳定，模型看到的场景相似。实现方式就是：**固定 query 中的显式边数 = 当前（噪声）图在 query 上的显式边数** \(k\)，这样本轮只重分配「哪 \(k\) 条为 1」，不增不减。

**实现与预期**：
- **当前实现**：\(k = k_{\text{noisy}}\) = 当前噪声图在 **本块 query** 上的显式边（argmax≠0）条数。Ideal 只优化「哪 \(k\) 条 query 为存在」，其余为 no-edge；训练目标 CE(pred, ideal) 让预测在 query 上的存在数也趋近 \(k\)。
- **多轮场景边数**：采样时每轮 merge 后，新图显式边数 = 旧图显式边数 − 本块 query 在旧图中的显式边数 + 本块 query 中预测为存在的边数。当「预测为存在的边数」= \(k_{\text{noisy}}\) 时，即与当前图在 query 上的显式边数一致，则 **新图显式边数 = 旧图显式边数**，即**同一扩散步 t 下各轮总边数保持不变**，场景相似。
- **结论**：该机制**可以实现预期**（多轮场景边数固定、面对的场景相似），前提是模型预测的 query 存在边数接近 \(k_{\text{noisy}}\)，而 CE(pred, ideal) 正是以 ideal（在 query 上恰有 \(k\) 条存在）为目标，因此训练会促使这一点。若模型系统性多预测或少预测存在，总边数会逐轮漂移，需通过 exist_pos_weight 等调节。

**配置**：`structure_ideal_fix_k: true`（默认）。由 `structure_only_global: true` 开启全图分支并构建 ideal。

**做法**：
- 记 \(k = k_{\text{noisy}} =\) 噪声图在 query 上的「存在」边数。
- 只优化 **哪 \(k\) 个 query 位置为 1**，其余为 0；不改变 query 上的总边数。
- 可微近似：用标量 \(s_i = k \cdot \mathrm{softmax}(\ell)_i \in [0,1]\)（\(\sum_i s_i = k\)），\(i\) 为 query 下标；ideal_soft 在 query 上为 \((1-s_i)\cdot \text{no-edge} + s_i \cdot \text{type}_i\)，对 \(\ell\) 做几步梯度下降最小化 \(L_{\text{struct}}(G(\text{ideal\_soft}), A)\)。
- 离散化：取 \(s\) 最大的 \(k\) 个位置置为「存在」（类型用真实图该位置类型或默认类型 1），其余置为 no-edge。

这样 ideal 的构建本身就是「只调 query、边数不变」，训练目标在边数上更匹配「多轮场景边数固定」。
