# IMDB配对包v1.1：恢复与Notebook交接

track=ENGINEERING，服务IMDB基础/CN/当前度相同、仅匿名角色读取贡献不同的配对实验。未开展科学训练或完整大图生成，无质量/公平效果结论。
修复审查基线bb32bfaa5b63f049cbcba4408e2184a39e5c8b6a；独立目录与分支，旧目录、ZIP、检查点不变。

## 已复现、已修复

新增回归在未修改实现上实际失败：第一次从recovery读取后，父latest写成裸文件名，第二次读取报existing checkpoint pointer/hash conflict。原失败XML保留。
新父指针保留recovery/相对路径，限制在永久目录及其recovery直接子目录，拒绝绝对路径、越界、符号链接、缺文件及hash/步数/payload冲突。
两处相同更新内容不同时拒绝；formal双臂模型/优化器库存与优化器更新计数被核验，间歇未参与参数不强求每步梯度。
滚动保存先发布新父指针再执行原两份保留策略，避免轮换删除旧目标使父指针悬空。

空间门区分首次与恢复：首次仍至少16GiB；恢复按实际计划永久文件、剩余滚动位、保守单文件上界、原子写入及STOP余量计算，并保留25%检查点余量和2.5GiB其他IO储备。
不会清理已有结果。实际输出路径的st_dev、挂载点/文件系统和可用空间记录在storage.json；持久性仍需操作者核实，路径名不作认证。
单文件估计是保守工程边界，不是磁盘配额或平台持久性保证；异常日志与任意多次额外STOP保存仍可能耗尽空间，门不足即停止。
preflight的570秒总门不变；超时/STOP记录incomplete、非零退出，不称算法失败，不自动formal。补充每臂临时检查点保存/恢复时间和字节数。

## 本地实际验证

原75项完整保留，新增17项恢复/空间/未完成状态回归，最终92通过、0失败、0跳过，pytest23.75秒，外层24.56秒，峰RSS937992KiB（约916.01MiB）。
5条warning均为继承测试record_property与JUnit xunit2格式的报告提示，不是模型告警。
新增回归含真正新Python进程恢复两个临时模块、独立AdamW、步数、全部CPU随机状态及后续实际输入摘要；不是只对内部loader做mock。

新manifest的公开preflight父入口→子进程→监督器CPU夹具通过：16.4826秒，峰RSS1417.453MiB；10次临时工程更新，科学更新0。
两臂CPU oracle对CPU target logits/p0/梯度最大差均0；同设备严格重复、保存恢复续算通过。不是CPU/GPU比较。
临时base检查点80352420字节，保存1.37046秒、恢复/加载0.99253秒；role80380294字节，保存1.49957秒、恢复/加载0.91922秒。
这些是本地文件系统/CPU数值工程结果，不能当Notebook成本。

实际依赖：Python3.9.23、Torch2.1.0、PyG2.3.1、NumPy1.26.4、SciPy1.11.0、NetworkX2.8.7、pytest7.4.4；单线程CPU，本票不调用GPU。
独立解包结果另见unpacked_acceptance.json，必须同一92项集合零跳过、载荷哈希与入口help/plan均通过，不能以旧75项替代。
交付闭包检查另发现构建脚本漏打无扩展名LICENSE，已修复并增加manifest载荷闭包断言；独立解包实际调用verify_package()，不只核对ZIP内已有文件。未发布草稿ZIP留存，不作为可用交付。

## 科学设置未改变

CPU prior复用与目标设备global零张量两补丁原样保留，Transformer/CN/当前度/角色编码/Newton/稳定CE/噪声/Bayes/查询/采样规则不变。
配置仅版本标识改变；normal2为双输入CE普通均值，是两臂共同新协议，区别历史IMDB单输入，非旧实验直接复现。
三对seed、每臂12000有效更新和96条T1000链仍只是待预算确认提案；不代表收敛或本轮执行授权。
base关闭目标读取和融合贡献，仍保留CN/当前度及同一联合头；共享公共初始张量、独立优化器、实际配对输入和随机流。

## Notebook待测与交接

没有访问Notebook/BW，没有执行GPU数值门、完整IMDB GPU更新或20块单步成本。本地此前CUDA可初始化也不代替模型梯度门。
历史GPU梯度超差记录保留，不放宽ATOL2e-6/RTOL1e-5；实际Notebook后端是否通过需新包对应预检。
环境/依赖缺失须停止并列明，不自动安装。t500前向加噪普通状态仅作成本初估，不保证自由链所有状态峰值和耗时。
中文HANDOFF_zh.md包含独立上传/校验/解包/环境/存储核实/预检/进度/STOP/证据导出，正式启动及恢复模板明确尚未授权。
固定回传PF/preflight.json、supervisor.json、launcher.log、environment.json、storage.json、numerical_gate.json及已完成cost_base/role.json；部分结果不得宣布通过。
发布后停止，接手助手先核实Notebook访问与环境并执行570秒预检，再依据两臂实际成本建议正式规模和预算。
