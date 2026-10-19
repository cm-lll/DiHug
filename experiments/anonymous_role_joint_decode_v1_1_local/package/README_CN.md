# 匿名联合角色度固定匹配MAP：本地文件接口小修版

独立目录anonymous_role_joint_decode_v1_1_local，固定基线2c1a2f2。
原v1源码、ZIP、权重和结果保留。无网络、SSH、BW、模型调用、训练、完整链、
自动下载或安装；本轮不重跑IMDB随机评分规模检查。

改动：正支持CDF采样；图位向量/编号独立交叉验收；严格本地冻结A1/B1标准读取器；
完整p0期望联合关系度到匿名角色的类型内一对一匹配；固定匹配MAP与离散诊断。
不是匿名全局MAP，不是IMDB条件随机采样器。度满足是约束保证，不是学习成果。

重要边界：标准接口通过明确标记的合成夹具，不等于旧真实导出格式已经接通。
本地只静态读取过旧导出器/合同/测试；真正A1/B1数组、原图和私有权重均未读取。
需要的字段、来源限制、旧格式缺口、匹配公式见LOCAL_INPUT_CONTRACT.md。
所有设置固定在LOCAL_INPUT_POLICY.json和原frozen_config.json。

## 本地CPU回归

使用已有NumPy/SciPy/pytest，不改变环境；具体实际版本见evidence/dependencies.json。

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
python verify.py --output /tmp/role_decode_local_tests_review01
python decode_frozen.py --help
```

verify隔离外部pytest筛选与父目录配置，核对本版完整node IDs且零跳过。
旧27项原样保留。合成CLI端到端测试确实经过父进程、子进程、读取器、匹配、求解、
评价和保存，不把内部函数或mock替代完整成功路径。

## 合成标准输入示例（不是实际模型概率）

```bash
python tests/fixture_export.py --output /tmp/role_decode_ENGINEERING_input01
```

该命令在创建夹具时写fixture_pins.json，记录新建合成文件的预期哈希，仅用于工程
复现；不能把这种自生成pin当作真实模型来源证据。正式输入应由另一侧按封存记录
准备标准工件，提供独立可信的预期SHA256。本机没有真实数据，不会自动寻找。

## 留给另一侧的显式真实入口，本轮未执行

下列变量均由调用者填写自己的本地工件和封存pin，不提供猜测路径，不自行计算当前
manifest哈希来充当可信预期值。每次只运行一个arm，不自动串联A/B。

```bash
: "${INPUT_MANIFEST:?填写本地标准manifest文件或根目录}"
: "${EXPECTED_MANIFEST_SHA256:?填写独立封存的manifest SHA256}"
: "${ARM:?A或B}"
: "${OUTPUT_DIR:?填写与输入和源码分离的新目录}"

python decode_frozen.py \
  --input "$INPUT_MANIFEST" --manifest-sha256 "$EXPECTED_MANIFEST_SHA256" \
  --arm "$ARM" --output "$OUTPUT_DIR" --max-seconds 300
```

若有相同arm/seed/来源的原无引导图，另用下面命令（不是自动后续命令），并指定
另一个新输出目录。原图缺失时报告比较缺失，不补抽。

```bash
: "${ORIGINAL_GRAPH:?填写对应原图NPZ}"
: "${EXPECTED_ORIGINAL_SHA256:?填写封存原图哈希}"
: "${ORIGINAL_RECEIPT:?填写原图来源回执}"
: "${EXPECTED_ORIGINAL_RECEIPT_SHA256:?填写封存回执哈希}"

python decode_frozen.py \
  --input "$INPUT_MANIFEST" --manifest-sha256 "$EXPECTED_MANIFEST_SHA256" \
  --arm "$ARM" --output "$OUTPUT_DIR" --max-seconds 300 \
  --original-graph "$ORIGINAL_GRAPH" --original-sha256 "$EXPECTED_ORIGINAL_SHA256" \
  --original-receipt "$ORIGINAL_RECEIPT" \
  --original-receipt-sha256 "$EXPECTED_ORIGINAL_RECEIPT_SHA256"
```

300秒包括启动、核验、匹配、求解、精确评价、保存及清理；CPU单线程。不再找一个
匹配、不调参、不换输入、不延时重试。不可行、超时或未认证最优均非零退出。

```bash
touch "$OUTPUT_DIR/STOP"
cat "$OUTPUT_DIR/controller.json"
cat "$OUTPUT_DIR/summary.json"
```

SIGINT/SIGTERM/SIGHUP/STOP清理本次自己的子进程组；不能捕获SIGKILL或机器故障。
成功图为decoded_graph.npz，所有比较/来源/数值界/时间/内存在summary.json。
不能只看summary.status；还需controller.exit_code=0、非空checks和图哈希。
partial状态不算完成，不可由一次MAP宣称大图多样性。本票交付后等待审查。
