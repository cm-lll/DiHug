# 第一阶段独立收尾

prefix_v2已保存八个科学终点与completion：30432机会、全部到达连通、科学1184.813秒；独立CPU验收已打印四条IMDB通过，随后进程以143结束。未收到用户停止指令，具体外部终止原因未确认，不能称质量失败或原验收完成。

单独CPU收尾重新验证八条全部轨迹、每步增量/覆盖/分量/选择及每32步特征、全部快照指标、CPU模型分数<=2e-5；不重生图、不重训、不改种子。先完整SHA核对原manifest，再冻结所有原trace/features/graph/report和源，独立验收上限2400秒。

收尾保存在prefix_finalized。原prefix_v2只新增interruption.json以及尚不存在的acceptance/summary：acceptance明确指向独立收尾及其SHA，并把收尾源和回执加入outputs依赖哈希。原manifest、completion、图、轨迹和源不覆盖。消费者据delegated acceptance继续既定确认，不把这次收尾计作新生成或训练。
