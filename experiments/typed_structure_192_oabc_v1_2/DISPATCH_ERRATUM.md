# PyG分派说明勘误与下一阶段边界

本次只静态核对已安装PyG2.3.1的utils/scatter.py，不运行网络、算子重复、更新、字典拟合或采样。

原README及封存tie_extrema.py的docstring中“Autograd's no-grad forward selects PyG's native reduction”不准确。
实际min/max分派条件为：

```python
if (not torch_geometric.typing.WITH_TORCH_SCATTER
        or not src.is_cuda or not src.requires_grad):
    # native scatter_reduce_, include_self=False
else:
    # torch_scatter.scatter(..., reduce='max'/'min')
```

grad mode与输入requires_grad并非同一个条件。custom Function.forward内部关闭自动微分记录，不等于输入requires_grad变False。
因此不能仅据no-grad断言训练前向换成原生归约。CPU与通常eval/no_grad产生的不需梯度输入会满足原生分支；若输入仍requires_grad，则仍按上面条件分派。
本勘误不声称逐次追踪了所有新调用。以前捕获的原算子分派证据与本次新函数数值验收仍各有原范围。

新函数的反向由自身backward定义，依赖精确极值掩码和计数，不依赖torch-scatter的并列argmax选择。
已完成的40次真实算子重复、六格同GPU梯度和恢复证据不因该说明错误失效；CPU/GPU失败也不删除或改判。

保留固定提交0c4ed6b4d2dfa5b67aa6983e7d665fbc49667d96、所有生产源码/初态/证据以及ZIP原字节和SHA256。
本分支只修正文档；源文件docstring的错误句亦以本勘误为准，不修改封存代码以免改变既有源码绑定。
本分支更新文档清单，修订哈希另列documentation_correction.json；历史固定提交和原ZIP清单原样保留，不把原ZIP说成已包含本勘误。

用户接受该聚合作为后续A/B/C公共基底。下一阶段限定同一GPU型号、软件环境和实现的配对实验，不要求先证明跨CPU/GPU逐元素等价，不宣称跨后端等价。
下一准备项为正式每类型4角色字典的训练材料、固定拟合/收敛规则、真实字典低/中/高噪声接口与成本，以及六次训练/12张完整图的完整预算。
冻结本轮聚合，不追加开销优化或模型模块。仍不启动正式字典拟合、训练或生成。
