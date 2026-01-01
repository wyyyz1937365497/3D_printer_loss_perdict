"""
3D打印机转角误差补偿系统配置文件
"""

# 数据生成配置
DATA_GENERATION_CONFIG = {
    'num_samples': 5000,  # 生成样本数量
    'max_velocity': 150,  # 最大速度 (mm/s)
    'min_radius': 5,      # 最小转角半径 (mm)
    'max_radius': 50,     # 最大转角半径 (mm)
    'nozzle_weight': 10,  # 喷头重量 (g)
    'noise_factor': 0.1,  # 噪声因子
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 64,
    'num_epochs': 300,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'validation_split': 0.2,
    'random_seed': 42,
    'hidden_sizes': [128, 256, 128],  # 神经网络隐藏层大小
}

# 模型路径配置
PATH_CONFIG = {
    'data_path': 'data/printer_displacement_data.csv',
    'model_path': 'models/displacement_predictor.pth',
    'scaler_x_path': 'models/scaler_X.pkl',
    'scaler_y_path': 'models/scaler_y.pkl',
    'results_plot_path': 'models/training_results.png',
}

# 3D打印机参数
PRINTER_CONFIG = {
    'max_print_speed': 200,  # 最大打印速度 mm/s
    'min_feature_size': 0.1,  # 最小特征尺寸 mm
    'positioning_accuracy': 0.01,  # 定位精度 mm
}