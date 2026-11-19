# R27表格导出修复和独立收尾

四图科学生成及全部路径重放后，CSV导出失败：IMDB另有孤点/目标覆盖字段，PubMed首行字段不能覆盖。原failure与部分CSV保留。新round27_finalize.py仅读取既有四轨迹，重新独立验收2048路径、64全图抽查、8快照和double分数，按字段并集输出CSV，缺少指标留空而不填零。不重跑生成/拟合，不修改原R27。结果在r27_finalize，来源和原summary/graphs/trace绑定，新的acceptance才允许后继。
