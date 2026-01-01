import torch
import numpy as np
import joblib
import pandas as pd
import os

# 导入模型定义
from train_model import DisplacementPredictor

def load_model_and_scalers(model_path='../models/displacement_predictor.pth'):
    """
    加载训练好的模型和标准化器
    """
    # 使用绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(project_dir, 'models')
    
    model_path = os.path.join(models_dir, 'displacement_predictor.pth')
    scaler_x_path = os.path.join(models_dir, 'scaler_X.pkl')
    scaler_y_path = os.path.join(models_dir, 'scaler_y.pkl')
    
    # 初始化模型
    model = DisplacementPredictor()
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # 加载标准化器
    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    
    return model, scaler_X, scaler_y

def predict_displacement(model, scaler_X, scaler_y, velocity, radius, weight, stiffness, ideal_x, ideal_y):
    """
    预测特定参数下的位移偏差
    """
    # 准备输入数据
    input_data = np.array([[velocity, radius, weight, stiffness, ideal_x, ideal_y]])
    
    # 标准化输入
    input_scaled = scaler_X.transform(input_data)
    
    # 转换为PyTorch张量
    input_tensor = torch.FloatTensor(input_scaled)
    
    # 预测
    with torch.no_grad():
        prediction_scaled = model(input_tensor)
        prediction_scaled = prediction_scaled.numpy()
    
    # 反标准化预测结果
    prediction = scaler_y.inverse_transform(prediction_scaled)
    
    return prediction[0]

def apply_correction(ideal_path, velocities, parameters):
    """
    对整个路径应用误差校正
    ideal_path: 理想路径点列表 [(x, y), ...]
    velocities: 每个点的速度列表
    parameters: 包含weight, stiffness的字典
    """
    # 加载模型
    model, scaler_X, scaler_y = load_model_and_scalers()
    
    corrected_path = []
    
    for i, (x, y) in enumerate(ideal_path):
        if i < len(velocities):
            v = velocities[i]
        else:
            v = velocities[-1]  # 如果速度不够，使用最后一个速度
            
        # 计算转角半径（简化计算，实际应用中需要更复杂的算法）
        if i > 0 and i < len(ideal_path) - 1:
            # 计算当前位置的曲率半径
            prev_x, prev_y = ideal_path[i-1]
            next_x, next_y = ideal_path[i+1]
            
            # 使用三点法估算曲率半径
            a = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            b = np.sqrt((next_x - x)**2 + (next_y - y)**2)
            c = np.sqrt((next_x - prev_x)**2 + (next_y - prev_y)**2)
            
            # 使用海伦公式计算三角形面积
            s = (a + b + c) / 2
            area = np.sqrt(s * (s-a) * (s-b) * (s-c))
            
            # 计算外接圆半径（近似曲率半径）
            if area > 0:
                radius = (a * b * c) / (4 * area)
            else:
                radius = float('inf')  # 直线段
        else:
            # 起始点和终点设为大半径（近似直线）
            radius = 1000.0
        
        # 确保半径在合理范围内
        if radius < 1:
            radius = 1
        elif radius > 1000:
            radius = 1000
        
        # 预测偏差
        displacement = predict_displacement(
            model, scaler_X, scaler_y,
            v, radius, parameters['weight'], parameters['stiffness'], x, y
        )
        
        # 应用校正（减去预测的偏差）
        corrected_x = x - displacement[0]
        corrected_y = y - displacement[1]
        
        corrected_path.append((corrected_x, corrected_y))
    
    return corrected_path

def test_model():
    """
    测试模型的预测能力
    """
    print("测试模型...")
    
    # 加载模型和标准化器
    model, scaler_X, scaler_y = load_model_and_scalers()
    
    # 测试几个样本
    test_cases = [
        {'velocity': 100, 'radius': 10, 'weight': 10, 'stiffness': 1.0, 'ideal_x': 5, 'ideal_y': 5},
        {'velocity': 50, 'radius': 20, 'weight': 10, 'stiffness': 1.2, 'ideal_x': 10, 'ideal_y': 10},
        {'velocity': 150, 'radius': 5, 'weight': 10, 'stiffness': 0.8, 'ideal_x': 15, 'ideal_y': 15},
        # 添加尖锐转角的测试案例
        {'velocity': 120, 'radius': 1, 'weight': 10, 'stiffness': 1.0, 'ideal_x': 20, 'ideal_y': 20},  # 尖锐转角
        {'velocity': 80, 'radius': 2, 'weight': 10, 'stiffness': 0.9, 'ideal_x': 25, 'ideal_y': 25},   # 尖锐转角
    ]
    
    print("预测结果:")
    for i, case in enumerate(test_cases):
        pred = predict_displacement(
            model, scaler_X, scaler_y,
            case['velocity'], case['radius'], case['weight'], 
            case['stiffness'], case['ideal_x'], case['ideal_y']
        )
        print(f"测试案例 {i+1}:")
        print(f"  输入参数: v={case['velocity']}, r={case['radius']}, "
              f"w={case['weight']}, k={case['stiffness']}")
        print(f"  理想位置: ({case['ideal_x']}, {case['ideal_y']})")
        print(f"  预测偏差: ({pred[0]:.4f}, {pred[1]:.4f})")
        print(f"  校正后位置: ({case['ideal_x']-pred[0]:.4f}, {case['ideal_y']-pred[1]:.4f})")
        print()

if __name__ == "__main__":
    # 确保模型目录存在
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(project_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 测试模型
    test_model()
    
    print("模型应用示例:")
    print("理想路径 -> 校正后路径，考虑转角误差补偿")
    
    # 示例路径校正
    ideal_path = [(0, 0), (5, 5), (10, 10), (15, 5), (20, 0)]
    velocities = [50, 100, 120, 100, 50]
    parameters = {'weight': 10, 'stiffness': 1.0}
    
    corrected_path = apply_correction(ideal_path, velocities, parameters)
    
    print("路径校正结果:")
    for i, (orig, corr) in enumerate(zip(ideal_path, corrected_path)):
        print(f"  点 {i}: ({orig[0]}, {orig[1]}) -> ({corr[0]:.4f}, {corr[1]:.4f})")