# 数量辅助与训练长度：CPU结果收尾

只读取已保存结果，不导入Torch、模型或科学worker。源版本为c3f9ed0；唯一旧文件修复为`finalize.py`的嵌套任务身份读取。旧失败回执保留，不覆盖旧包或旧结果。

主面板：E-earlyCN，IMDB2911/2931、PubMed2912/2932，CE/CE_COUNT，EMA0.99，两个采样seed2026102821/22。数量公式沿用3be64f7，lambda=0.001。

合计48张原图：3000和6000各16张，PubMed1000/2000补充16张。LCC另列48份；IMDB没有1000/2000生成，不用固定验证p0替代。

执行（独立新目录，data2临时空间，线程1）：

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/data2/lyh/gnn_project/DiHug/results/count_aux_closeout_cpu_tmp
python -B build.py --output /data2/lyh/gnn_project/DiHug/results/NEW_CLOSEOUT
python -B report.py --output /data2/lyh/gnn_project/DiHug/results/NEW_CLOSEOUT
python -B audit.py --output /data2/lyh/gnn_project/DiHug/results/NEW_CLOSEOUT
python -B -m unittest test_closeout -v
python -B package.py --output /data2/lyh/gnn_project/DiHug/results/NEW_CLOSEOUT
```

完整包有可独立运行的`recompute_cpu.py`，只需现有NumPy/NetworkX，不安装依赖。运行会另建CPU复算结果，不能覆盖原结果。

查看顺序：SUMMARY_ZH.md、REPORT.md、METRIC_DIRECTORY.md、完整tables和evaluation.json。逐图误差先算后平均，配对和长度变化分开。不能把数量项胜过同期CE写成其自身持续改善。

图和权重按任务身份、采样seed、对应EMA文件及SHA绑定；大体积科学权重只登记，不发布。旧科学完成时间、失败、账本和截止不改；本次修复和发布时间另记。不授权训练、前向、生成或完整图任务。
