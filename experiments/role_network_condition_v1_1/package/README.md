# 匿名联合角色条件v1.1：先审查，后执行

本次仅交付代码及CPU回归；完整面板尚未启动。用户已将面板累计上限改为90分钟，但要求审查通过后才执行。除版本/时限外，科学配置与v1逐项一致。

仅增加输入残差，不调整输出概率、不匹配角色、不修图。A=原CN，B=A+完整Xt度特征，C=B+匿名联合角色注意力。详见发行目录DESIGN.md及固定config.json。

CPU依赖：Python3.9.23、Torch2.1.0、PyG2.3.1、NumPy1.26.4、SciPy1.11.0、NetworkX2.8.7、pytest7.4.4。没有自动安装逻辑；不应覆盖设备厂商Torch。

```bash
cd dihug_role_network_condition_v1_1/package
env -u PYTEST_ADDOPTS CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest tests -q

# 仅在代码审查通过后执行：先成本门限，再运行完整冻结面板。
# 输出目录及同名.budget.json必须不存在，且在源码目录之外。
python run.py --output /tmp/role_condition_new_run

# 安全停止自己启动的任务，保留未完成证据：
touch /tmp/role_condition_new_run/STOP
```

退出0=完整面板完成，3=成本门限不通过（不是学习失败），124=父监督超时，130=手动停止；其他非零=失败。摘要与控制器回执均须检查。父入口包括启动、成本测试、训练、采样和评价，总计算上限5275秒，清理最多5秒，保守预留120秒，合计不超过5400秒。不会自动重跑或延长。

启动前以排他方式创建输出目录旁的.budget.json，绑定第一次激活时间、5400秒截止时间与源码manifest哈希。已存在的账本或输出拒绝覆盖。此最小原型不支持学习面板断点恢复；重启拒绝，不会重新分配90分钟。保留部分证据，不能用新目录自动重试绕开累计预算。

学习、验证及实际生成曲线只有完整面板运行后才有；本次等待审查，没有这些曲线或科学生成样本。原v1成本门限失败及24工程更新记录仍保留在旧版本；本版evidence为新增正确性实测，不能当作BW或生成质量结果。原role_degree文件只为保留旧核心回归依赖，新入口不调用其中引导。
