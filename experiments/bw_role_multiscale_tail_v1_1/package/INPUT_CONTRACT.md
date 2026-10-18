# v1.1：真实旧诊断只读转换合同

已读取用户提供的 joint_role_frozen_diag_v1.zip，SHA256：
c0fc3a8dade80a66f550a0506e405bf62bf80b3c1fb9f902d6a8baebf861194b。
实际六状态数组、collection、Genre报告和私有权重仍不在本地；真实转换尚未执行。
不调用模型，不反序列化私有检查点，不补跑链，不生成替代输入。

## 旧产物实际接口

export-legacy接受原ZIP、实际旧结果根目录、用户核实的evidence_hashes.json SHA256。
扫描collection.json，从最近祖先provenance.json的arm和collection.time识别案例，
不依赖A/X0500等目录名称。必须恰好A/B×t=500/20/1，拒绝重复、缺失和异常案例。
核对原结果ledger、activation、completed、源码release和各arm provenance。
完整原轨迹和最终6000更新检查点路径读取activation.states/checkpoints及ZIP原始pins，
流式哈希所有被引用文件，不猜检查点时间戳、不加载模型；原文件缺失就拒绝。
新输出必须位于旧结果树以外，且目录事先不存在。

旧baseline.npz实际含state,p0,q0,types,pairs,prior,alpha等；
collected.npz含state,p0,coverage。q0映射为q之前用原MarginalKernel.reverse
重算下一时刻s=t-1后验，最大绝对差<=5e-14；p0不是q。
两份state/p0必须精确相同，并符合collection中shape/dtype/原始字节摘要。
types/pairs/prior/alpha必须与固定IMDB参考和原T1000日程完全一致。

canonical slots/families从同一原TypedSlots实现确定性重建；
t从核对过的collection.time重建；query_block从实际旧worker规则重建。
原源码worker.py查询规则为plan_for(case,[2026101701,t,61],20)。
依块号0..19，将ids.tobytes()、families.tobytes()、query.tobytes()依次加入同一SHA256，
不加入块编号、长度等新封装字节。完整摘要必须精确等于collection.query_sha256，
同时核验覆盖、原coverage和全部合法槽顺序。仅20块/相同槽数量不构成证据。

field_origins对state/p0/q/types/prior/alpha写original（q原字段q0）；
slots/families/t/query_block写deterministically_reconstructed并保存配方。
legacy_source_contract.json和producer_config.json是从原ZIP确定性导出的新收据，
不是声称旧目录里本来就有的独立manifest。转换源码、原文件与新文件分别保存哈希。

## 新中间合同

six_states.json的format=legacy_export_v1_1，base_commit仍为原始状态生成基线
acbbddb5cdbcbaa2ae129bcd59e434e6a298b130（不是本修复源码版本）。
reference_sha256=10f1dbbbe55c6a069594b2dcb590d1b8295ca5436320ef4597cd30e6b7f73e42。
records恰好六项，每项：arm,t,reference_sha256,npz,npz_sha256,metadata,metadata_sha256,
checkpoint/checkpoint_sha256,source_manifest/source_manifest_sha256,config/config_sha256。
路径可相对manifest目录或绝对路径。

NPZ：state,p0,q,slots(2,S),families(S),types(N),prior(R),alpha(1001),t标量,
query_block(S，活跃槽0..19，确定性空族-1)。不删除空族。
metadata保存原生A/B、6000更新最终权重、来源/配置/数组哈希、重建来源、完整查询摘要、
原collection/Genre报告及全部原输入哈希。short开始和结束都重新核验原输入；
最后重新哈希、评价和保存也受STOP和相对预算检查，未完成不能标complete。

## 旧独立评价对接

只读genre_baseline.json的嵌套independent_diagnostic_frequencies.omega，
原形状(64,)，明确嵌入完整(64,R)的列8；Movie=2、Genre=3、pairs[8]=[2,3]。
其他列为0，不重新生成评价频率；保存原字段路径、报告文件哈希和关系映射。
该银行不参与优化目标构造。

CF_normalizer=1，目标字段independent_diagnostic_frequencies.mean_squared_mean_cf_distance：
期望经验CF相对于前向加噪目标的平方误差，不加条件方差，
不得替换成mean_expected_squared_empirical_cf_distance。
W1字段w1_between_expected_empirical_laws：完整原始整数PMF均值之间、单位间隔未缩放W1，
不是期望度直方图，也不是生成图W1的期望；不套用旧描述表的PMF归一化。

每个baseline必须先复算旧Genre CF及W1并对齐，才允许新版引导。
预先冻结接口数值容差：CF绝对2e-10+相对1e-9；
W1绝对1e-8+相对1e-9，加既有FP64操作量allowance。
allowance不是严格区间证书；对齐失败就停止，不改定义。
这不是改变PREDECLARED_POLICY.json中的科学判据。

## 成本与停止

cost入口只运行12节点/66槽确定性合成工程概率，另读取固定IMDB参考核算运算量；
不读取/制造真实六状态，不输出质量结论。分开冷启动、目标、后验、梯度、回溯、评价、IO和RSS。
精确PMF的同一目标按frozen law/time/type/channel缓存；逐节点DP和全部预声明指标保留。
缓存前后小图结果逐字节一致，未改为近似、抽槽或删指标。

每次控制器调用10..900秒，从入口导入开始计时，包含核验/计算/评价/最终哈希/保存/清理；
工作提前12秒停止，父进程提前6秒进入清理，不沿用旧绝对截止。
STOP、SIGINT、SIGTERM、SIGHUP均停止并清理自建进程组；不杀其他任务。
SIGKILL、机器掉电不可捕获；不能保证finally执行。工作进程禁止脱离自建进程组。
父进程由SIGKILL直接杀死时，不能保证子进程自动回收；须检查controller_child.json及进程。
完整性缺失、超时或清理失败均非零退出，部分输出只标incomplete/stopped/failed。
缺真实源文件不回退示例；不会自动启动任何训练、科学短测或完整链。
