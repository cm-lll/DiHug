# 已授权的八卡正式小图实验

本轮承接固定42b83015c2c32e5f2ae57423301a662081c622e5。用户明确授权仅修资源门后启动，无需重新全面审查。旧版本README中的“未授权”是历史准备状态，不代表本次授权。

唯一既有package修改是launch.py：GPU利用率记录为遥测，不作为单独拒绝条件；必须有用户对所选八张UUID的明确授权、无已知调度预留冲突、无其他计算任务/设备打开句柄，并满足显存要求。最低空闲5556MiB来自旧最大reserved3508MiB加2048MiB余量，不承诺自由生成所有状态的峰值。7项纯CPU资源门样例通过，没有重跑八格GPU。

权限依据是用户本条八卡授权，不是“进程列表为空”。本地未发现调度器工具/进程、作业/设备隔离环境变量、计算进程或GPU打开句柄；设备信息与原已核验2080Ti一致。没有更改驱动、依赖、权限、隔离或他人任务。

正式激活入口activate_once.py仅可首次执行。它写入唯一8400秒账本和本条授权的绑定回执，然后exec已审launch.py。已有ledger/approval/run时拒绝重启；不重新授时。协议文件的旧准备预算注释原字节保留，实际本次执行预算由新ledger和explicit effective_execution明确覆盖，不倒改旧账本。

实际激活：2026-10-28 09:06:47.931612 Asia/Shanghai；硬截止11:26:47.931612。保留既有截止前1200秒停止训练/采样的交付余量。准备时间另见formal_preparation/completed.json。

科学范围不变：IMDB/PubMed各192节点；O无CN/ReFeX/RolX，A加CN，B为630维ReFeX无CN，C再加固定RolX系数；保留共同当前Xt度分支。K分别5/4/3/2和8/8/8/6，PubMed全部命中上界限制保留。不拟合字典、不加载工程权重。每格1000次normal2、全槽一次前向；0/500/1000保存，只有1000用于生成。IMDB训练seed2026102811，PubMed2026102812；每模型采样seed2026102821/2026102822，16条T1000，无修图/补抽。

一格一卡的固定LPT分工（UUID完整记录在frozen_schedule.json）：卡0 PubMed A；1 PubMed C；2 PubMed B；3 PubMed O；4 IMDB A；5 IMDB C；6 IMDB B；7 IMDB O。CPU每进程1线程。原预测135.05分钟含未测并发余量，不保证按时完成。

最终package源码绑定：6349f816a273679e2f08bb0d887065d23f77b488e68b3a5aca036683d6258429。
完整文件及运行授权绑定见formal_preparation/final_source.json和正式结果control/approval.json。根summarize.py未变。资源补丁见formal_preparation/resource_gate.patch。

效果必须以原始生成图判断；incident4为W1主口径，all10为兼容。原值/绝对参考误差/分布误差分别汇总；null和失败不抹去。单训练seed/数据集、两个生成seed只支持探索性观察，不是稳定收益、显著性或跨图泛化证据。

运行期间不得执行activate_once.py第二次，不自动重新训练或补抽。安全停止为在本次run目录创建STOP；正式执行已获授权，本文件不是新的审批请求。
