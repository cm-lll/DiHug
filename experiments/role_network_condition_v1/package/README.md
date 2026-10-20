# 匿名联合角色条件：本地CPU最小版本

仅增加输入残差，不调整输出概率、不匹配角色、不修图。A=原CN，B=A+完整Xt度特征，C=B+匿名联合角色注意力。详见发行目录DESIGN.md及固定config.json。

CPU依赖：Python3.9.23、Torch2.1.0、PyG2.3.1、NumPy1.26.4、SciPy1.11.0、NetworkX2.8.7、pytest7.4.4。没有自动安装逻辑；不应覆盖设备厂商Torch。

```bash
cd dihug_role_network_condition_v1/package
env -u PYTEST_ADDOPTS CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest tests -q

# 仅在另行授权运行时执行：先成本门限，再决定是否运行完整冻结面板。
# 输出目录必须不存在，且在源码目录之外。当前机器预计成本门限将返回3。
python run.py --output /tmp/role_condition_new_run

# 安全停止自己启动的任务，保留未完成证据：
touch /tmp/role_condition_new_run/STOP
```

退出0=完整面板完成，3=成本门限不通过（不是学习失败），124=父监督超时，130=手动停止；其他非零=失败。摘要与控制器回执均须检查。父入口包括启动、成本测试、训练、采样和评价，总计算上限1675秒，清理最多5秒，另为两次正确性复验预留120秒，合计不超过1800秒。不会自动重跑或延长。

学习、验证及实际生成曲线只有完整面板运行后才有；本次成本门限未通过，没有这些曲线或科学生成样本。evidence内costs/summary/controller为实际CPU测量，不能当作BW或生成质量结果。原role_degree文件只为保留旧核心回归依赖，新入口不调用其中引导。
