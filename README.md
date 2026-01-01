# 3D打印机转角误差补偿系统

本项目旨在通过MATLAB仿真生成3D打印转角误差数据，并使用Python训练神经网络模型来预测和补偿这些误差，从而提高打印质量。

## 项目结构

- `matlab_simulation/` - MATLAB仿真脚本
- `data/` - 存储生成的数据集
- `python_scripts/` - Python数据处理和训练脚本
- `models/` - 训练好的模型

## 工作流程

1. 使用MATLAB仿真生成转角误差数据
2. 使用Python训练神经网络模型
3. 模型用于预测和补偿打印误差