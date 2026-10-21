# Notebook交接：仅预检，正式实验尚未授权

本版修复基线bb32bfaa5b63f049cbcba4408e2184a39e5c8b6a，使用新的源码manifest、config版本及ZIP。
Sol只在本地CPU测试；Notebook环境、GPU数值门及完整IMDB成本均待接手助手实际检查，不能沿用旧包证书。
不要安装/升级Torch、驱动、PyG或编译扩展，不修改代理、安全设置或设备可见范围，不传账号密码。

## 上传、可信校验、独立解包

从固定审查提交同时取得dihug_imdb_role_pair_gpu_v1_1.zip和外部SHA256SUMS。
以交付回复的固定提交与ZIP哈希为来源；不要把当前ZIP自算哈希当成可信预期值。
上传至/root/private_data/gnn_projects中的新交接目录；该路径仅为示例，持久性须通过平台说明确认。

```bash
cd /root/private_data/gnn_projects
# 先确认ZIP与SHA256SUMS来自本次固定审查提交。
sha256sum -c SHA256SUMS
test ! -e imdb_role_pair_gpu_v1_1 || { echo '目标已存在，拒绝覆盖'; exit 1; }
unzip -n dihug_imdb_role_pair_gpu_v1_1.zip
cd imdb_role_pair_gpu_v1_1
sha256sum -c SHA256SUMS
cd package
PYTHON_BIN=/usr/local/bin/python3.11
test -x "$PYTHON_BIN" || { echo '缺少指定Python，不自动安装'; exit 1; }
"$PYTHON_BIN" - <<'PY'
import sys,importlib.metadata as m
print(sys.version,sys.executable)
for name in ('torch','torch-geometric','numpy','scipy','networkx','pytest'):
    print(name,m.version(name))
import pair_core as p
print('package_manifest_sha256',p.verify_package())
PY
```

缺失依赖时保留原报错，停止预检并回传，不自动安装。HIP平台使用torch.cuda API正常，不要求torch.version.cuda非空。

## 存储与前台预检

```bash
RUN_PARENT=/root/private_data/gnn_projects/imdb_role_pair_gpu_v1_1_runs
mkdir -p "$RUN_PARENT"
df -h "$RUN_PARENT"
findmnt -T "$RUN_PARENT"
# 接手助手另核对平台说明确认此挂载卷持久性和剩余容量，不能只看目录名字。
PF="$RUN_PARENT/preflight_01"
test ! -e "$PF" || { echo '预检输出已存在，拒绝覆盖'; exit 1; }
PYTHON_BIN=/usr/local/bin/python3.11 bash run.sh preflight --device cuda --output "$PF"
```

前台运行便于确认启动；终端会打印START/END，详细进度在launcher.log。
570秒从轻量父进程激活开始，包含Torch导入/设备初始化/核验/临时更新/成本测量/保存，清理额外至多10秒单列。
超时或STOP是未完成，不是模型故障；失败/未完成均非零退出且不自动进入正式训练。
小图先以严格模式检查CPU/GPU前向、CE、参与梯度、一次更新、重复与保存恢复，维持ATOL2e-6/RTOL1e-5。
门通过后普通模式完整IMDB每臂5次临时normal2更新（1预热、3稳态、1trace），每臂2次同输入完整20块单步，非连续链。
工程权重不是科学训练，不进入正式初始化。Newton/度似然留CPU，网络及学习模块使用所分配可见逻辑卡0。
不要把CPU夹具或本地CUDA可初始化当作NotebookGPU通过。旧GPU梯度失败历史保留，不外推另一后端必失败。

另开终端，进入同一个package目录：

```bash
PF=/root/private_data/gnn_projects/imdb_role_pair_gpu_v1_1_runs/preflight_01
tail -f "$PF/launcher.log"
# 或随时查看（不启动新任务）
PYTHON_BIN=/usr/local/bin/python3.11 bash run.sh progress --output "$PF"
# 安全停止，仅停止本次入口的子进程组，不影响其他任务。
PYTHON_BIN=/usr/local/bin/python3.11 bash run.sh stop --output "$PF"
```

preflight不续跑到formal，也不提供预检中途续跑；未完成目录保留，后续重新预检须显式新目录。formal的原批准总预算恢复不重置。

## 固定回传位置与清单

同一个PF目录：preflight.json是最终摘要；supervisor.json是外层退出/时间/回收回执；activation.json保存570秒起点；process.json记录本次PID与命令。
environment.json为实际依赖/设备；storage.json记录实际输出挂载卷/空间估算；numerical_gate.json为两臂数值证据。
完整通过还应有cost_base.json、cost_role.json，逐臂更新/20块单步/保存成本、CPU峰RSS和GPUallocated/reserved/device占用；cost_*_progress.json保留部分进度。
失败或超时保留已存在文件、launcher.log以及failed_reads，不因缺项声称完成。gate_base/gate_role内仅临时工程检查点，交接摘要通常不需要上传权重字节。

在父目录导出小型证据包，不包含权重和任何账号文件：

```bash
cd /root/private_data/gnn_projects/imdb_role_pair_gpu_v1_1_runs
test ! -e preflight_01_evidence.tar.gz || { echo '导出文件已存在'; exit 1; }
tar --exclude='*.pt' --exclude='*.pth' -czf preflight_01_evidence.tar.gz preflight_01
sha256sum preflight_01_evidence.tar.gz
```

回传上述tar.gz、终端退出码，以及preflight.json、supervisor.json的正文。助手先判定新包实际GPU数值门和完整成本是否都完成，再提出正式规模/预算建议。
t500普通前向加噪状态只用于初估，不能保证自由生成所有状态的耗时/显存；三个稳态更新的统计很有限。超时不能把部分成本外推成完整双臂测量。

## 科学方案与未授权命令

两臂同Transformer、CN、当前Xt度分支、联合头，base只关闭匿名角色读取与融合贡献；公共张量逐个复制、独立优化器、同配对输入及随机计划。
normal2每步两份普通加噪CE的均值，是本轮共同协议，区别历史IMDB单输入，不称旧训练的直接复现。
三对训练seed/每臂12000更新/96条T1000完整链只是待预算确认提案，不是执行授权或收敛保证。
正式方案无角色硬匹配、输出概率控制、度配额或修图。

以下仅展示接口，尚未授权执行；不得自动填写预算并运行：

```bash
# 只输出冻结提案，不训练。
PYTHON_BIN=/usr/local/bin/python3.11 bash run.sh plan
# 尚未授权：经用户另行确认冻结提案哈希、总预算及持久存储后才可执行。
PYTHON_BIN=/usr/local/bin/python3.11 bash run.sh formal \
  --preflight /root/private_data/gnn_projects/imdb_role_pair_gpu_v1_1_runs/preflight_01 \
  --output /root/private_data/gnn_projects/imdb_role_pair_gpu_v1_1_runs/formal_01 \
  --confirmed-budget-seconds USER_CONFIRMED_SECONDS \
  --approved-plan-sha256 USER_CONFIRMED_PLAN_HASH --persistent-output-confirmed
# 尚未授权：只恢复本次新版正式目录，保留原预算与来源；不是旧版迁移入口。
PYTHON_BIN=/usr/local/bin/python3.11 bash run.sh formal --resume \
  --preflight /root/private_data/gnn_projects/imdb_role_pair_gpu_v1_1_runs/preflight_01 \
  --output /root/private_data/gnn_projects/imdb_role_pair_gpu_v1_1_runs/formal_01 \
  --persistent-output-confirmed
```

正式恢复前须确认旧进程退出，并显式归档STOP（不得删除来源或重置activation）。恢复检查源码/配置/数据/依赖及检查点，不承诺普通GPU跨中断数值轨迹逐位相同。
永久与滚动检查点会核对内容digest、文件SHA256、步数、两臂与优化器状态；目录越界/缺文件/冲突即拒绝。
首次16GiB门与恢复剩余需求分开；详细保留策略和保守公式见package/README_zh.md，storage.json含实际输出卷信息。不会自动清理已有结果。

发布后Sol停止；接手助手只执行Notebook预检，尚无正式训练、完整生成或PubMed执行授权。
