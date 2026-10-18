# 真实六状态短测输入合同

现有六状态与原独立64频率工件本地缺失。不得重新调用模型、跑链或生成频率补齐。
需要从已有产物只读导出以下字段；所有SHA由用户对实际文件确认。
元数据交叉核对不是证明生产者一定按声明执行的数学证书。

six_states.json字段：base_commit=acbbddb5cdbcbaa2ae129bcd59e434e6a298b130；
reference_sha256=10f1dbbbe55c6a069594b2dcb590d1b8295ca5436320ef4597cd30e6b7f73e42；
records恰好六项，arm=A/B，t=500/20/1。每项字段：

```
arm,t,reference_sha256,
npz,npz_sha256,metadata,metadata_sha256,
checkpoint,checkpoint_sha256,
source_manifest,source_manifest_sha256,config,config_sha256
```

路径相对manifest所在目录或绝对路径。checkpoint/source_manifest/config必须是原生产产物，
脚本只读流式哈希，不加载模型、不上传权重。

每个原状态NPZ需要：
- state：完整Xt的二元状态。
- p0：该Xt原查询计划得到的干净边概率；q：原后验产生的下一时刻概率，不能混称。
- slots(2,S)、families(S)、types(N)、prior(R)、alpha(1001)、t标量。
- query_block(S)：原20块各槽所属块0..19，非活跃槽为-1。
均为完整合法上三角槽，顺序与原TypedSlots全局ID精确一致。

原metadata JSON需要：
arm,model_arm(A/B),t,T=1000,query_rounds=20,sampling_seed,
reference_sha256,checkpoint_sha256,source_manifest_sha256,config_sha256,
completed_updates=6000,final_weight=true，以及array_sha256映射，涵盖全部上述NPZ数组。
数组摘要使用：
```
signature({"shape":list(a.shape),"dtype":str(a.dtype),
           "raw_sha256":sha256(a.tobytes()).hexdigest()})
```
signature沿用stepwise_role.core的排序键紧凑JSON SHA256。
若旧工件未存某字段，应先列出缺失及可恢复证据，不伪造“原本已保存”。
原q重新计算允许最大绝对差5e-14；20块计划按原[seed,t,61]重构核对。

原独立评价spec JSON需要：
reference_sha256、profile_sha256（匿名完整角色profile的signature）、Genre_type=3，
omega64(64,R)、CF_normalizer(1或4，必须沿用原值)、
CF_contract="squared_distance_of_expected_empirical_joint_CF_to_forward_target"，
W1_contract="W1_of_expected_empirical_integer_degree_PMF_unscaled"，
original_artifact_source（相对spec路径或绝对路径）、original_artifact_sha256、frequency_array_key。
原频率工件可为JSON/NPZ；spec数组须与其完全一致。不得提供新生成的评价银行。
若旧诊断定义与这里不同，入口停止；需补齐原定义再适配，不声称已经复现旧指标。

输出不生成图，仅保存冻结目标、输入哈希、逐状态前后指标、控制器/计时和完整性状态。
没有任何“缺输入即回退示例图”“只算通过项”或新增模型调用路径。
