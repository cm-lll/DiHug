# 依赖、资产和复现边界

本分支用于逐行审查实际执行源码。没有重新运行模型；语法、文件哈希、schema容量及依赖导入检查不代替完整科学复现。

## 环境

已使用 Python 3.9.23，Torch 2.1.0+cu121，PyG 2.3.1，torch-scatter 2.1.2+pt21cu121。版本见 `requirements.txt`。CUDA扩展需与Torch/CUDA ABI匹配，不能无条件用最新版本。环境未安装原生HGEN所需的powerlaw、igraph、numba、torch-two-sample；评价代码提取少量已冻结函数并明确保留不可用项。`lcc_eval/vendor` 的原文件只作为函数来源，不能直接运行所有历史脚本并假定其旧工程依赖存在。

CPU审查命令（不加载模型或数据、不占GPU）：

```bash
python tools/audit_source.py
```

## 未包含的运行资产

1. 两张192节点诱导子图：各自 `types`、无向规范边、1起始关系标签、schema；原始NPZ哈希在 `configs`，文件本身未上传。
2. 原始角色字典NPZ、选择记录及扰动面板。K与字典哈希见 `configs/dictionaries.json`；不能重新拟合后冒充同一个封存字典。
3. 配对的科学theta0、输入哈希合同、完整随机状态、模型/EMA/AdamW检查点。仅相同整数seed不足以替代这些状态。
4. 原服务器票据账本、资源授权与审批。**本快照不提供可激活审批**，不关闭任何绑定或预算检查。

实际运行入口见README。源码中的 `OUT`、`TASKOUT`、`BASELINE`、`SCI` 和旧时间戳是票据绑定；因此launch/worker不是独立的通用CLI。审查者需在单独工程中配置自己的新输出目录、外置资产和明确的运行审批；不应伪造旧回执或删除哈希校验。必要的路径适配必须另存并更新绑定，不能称为与原始文件逐字节相同。

本次仅对拷贝中发现的个人部署路径作占位替换：源解释器路径替换为 `/opt/di-hug-env/bin/python`（公开占位值），源码根路径如有则替换为 `/opt/di-hug-source`。这不是已验证的可迁移适配；D历史入口仍需指定真实解释器。每文件原始/发布哈希及 `path_sanitized` 记录在 `provenance/source_files.json`，科学公式和原运行树未改。历史配置中的 `engineering_only`、`formal_authorized`、normal2及20块字段属于底层来源元数据；**实际训练设置以运行目录生成的配置为准**，不能直接执行底层历史main。

## 版本与历史

本审查分支为不继承旧仓库对象的独立快照，审查分支完整可达历史只有本次源码提交；主仓库和现有实验分支不改写、不删除。实际科学代码版本由原运行时source manifest与逐文件SHA绑定，不能以发布分支的新commit冒充历史训练commit。来源清单记录当前工作仓库定位commit及各文件属于哪些实际运行清单。

早CN、历史normal2、BW预检及D/ARD适配文件保留只为解释条件路径和依赖。正式构造及执行配置已在README逐一标注；这些代码的存在不表示所有候选被采用或在完整图上通过。`package/refit_dictionary.py` 记录原字典50000迭代补拟合流程，包含当时固定绝对截止；仅作来源审查，不是当前有效重拟合授权或可直接复用的计时入口。
