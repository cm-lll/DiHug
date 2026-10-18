# B1-only 原接受步归因侧车

固定基线7049e64；全部121个原包文件逐字节不改。新增代码在package/b1_attribution。
package/B1_DESIGN.md给出完整公式、坐标、W1阈值区间、数值容差和结论边界。
package/B1_COMMANDS.md给出手动检查/900秒诊断命令及缺失的两项封存SHA要求。
package/evidence_b1为本轮工程证据；所有继承evidence目录均属于旧版本。
package/v1_2_to_b1_sidecar.diff是相对固定v1.2的纯新增补丁。

本地原133+新增20项CPU检查通过；只在确定性小图上创建旧控制器工程夹具并核验重放，
真实入口没有guide.apply/候选搜索，仅重建原case_05记录的那一个接受步。
真实notebook输入不可访问，没有读取真实B1数组或复算真实指标，无BW、GPU、训练或图采样。
当前只交代码审查；实际运行还需要封存summary及case_05 SHA256，不能用inspect现场值冒充来源证明。

产物、公式及命令请同时审查。小图梯度通过不代表真实B1复现或一阶解释已获证实。
