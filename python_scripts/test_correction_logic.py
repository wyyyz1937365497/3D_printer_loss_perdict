import torch
import numpy as np
import pandas as pd
import joblib
import os
from train_model import DisplacementPredictor

def test_correction_logic():
    print("测试校正逻辑...")
    
    # 修复路径问题
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'printer_displacement_data.csv')
    df = pd.read_csv(data_path)
    
    # 随机选择一些样本进行测试
    sample_indices = np.random.choice(df.index, size=10, replace=False)
    sample_data = df.iloc[sample_indices]
    
    print(f"测试样本数: {len(sample_data)}")
    
    # 加载模型和标准化器
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(project_dir, 'models')
    
    model_path = os.path.join(models_dir, 'displacement_predictor.pth')
    scaler_x_path = os.path.join(models_dir, 'scaler_X.pkl')
    scaler_y_path = os.path.join(models_dir, 'scaler_y.pkl')
    
    # 初始化并加载模型
    model = DisplacementPredictor()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # 加载标准化器
    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    
    # 提取特征和真实偏差
    feature_columns = ['velocity', 'radius', 'weight', 'stiffness', 'ideal_x', 'ideal_y']
    X = sample_data[feature_columns].values
    y_true = sample_data[['displacement_x', 'displacement_y']].values
    
    # 标准化输入
    X_scaled = scaler_X.transform(X)
    
    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X_scaled)
    
    # 预测
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).numpy()
    
    # 反标准化预测结果
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # 计算误差
    prediction_errors = np.sqrt((y_true[:, 0] - y_pred[:, 0])**2 + (y_true[:, 1] - y_pred[:, 1])**2)
    print(f"平均预测误差: {np.mean(prediction_errors):.4f}")
    
    # 获取理想位置和实际位置
    ideal_x = sample_data['ideal_x'].values
    ideal_y = sample_data['ideal_y'].values
    displacement_x = sample_data['displacement_x'].values
    displacement_y = sample_data['displacement_y'].values
    
    # 实际位置 = 理想位置 + 位移偏差
    actual_x = ideal_x + displacement_x
    actual_y = ideal_y + displacement_y
    
    # 计算原始误差（实际位置与理想位置之间的距离）
    original_errors = np.sqrt((actual_x - ideal_x)**2 + (actual_y - ideal_y)**2)
    
    # 尝试校正：从实际位置减去预测的偏差
    corrected_x = actual_x - y_pred[:, 0]  # 实际x - 预测偏差x
    corrected_y = actual_y - y_pred[:, 1]  # 实际y - 预测偏差y
    
    # 计算校正后的误差
    corrected_errors = np.sqrt((corrected_x - ideal_x)**2 + (corrected_y - ideal_y)**2)
    
    print(f"原始平均误差: {np.mean(original_errors):.4f}")
    print(f"校正后平均误差: {np.mean(corrected_errors):.4f}")
    print(f"误差改善: {((np.mean(original_errors) - np.mean(corrected_errors)) / np.mean(original_errors) * 100):.2f}%")
    
    print("\n详细比较（前5个样本）:")
    print("样本\t理想位置\t\t实际位置\t\t预测偏差\t\t校正后位置\t\t理想-实际\t理想-校正")
    for i in range(min(5, len(sample_data))):
        orig_err = np.sqrt((actual_x[i] - ideal_x[i])**2 + (actual_y[i] - ideal_y[i])**2)
        corr_err = np.sqrt((corrected_x[i] - ideal_x[i])**2 + (corrected_y[i] - ideal_y[i])**2)
        print(f"{i}\t({ideal_x[i]:.2f}, {ideal_y[i]:.2f})\t({actual_x[i]:.2f}, {actual_y[i]:.2f})\t"
              f"({y_pred[i, 0]:.2f}, {y_pred[i, 1]:.2f})\t({corrected_x[i]:.2f}, {corrected_y[i]:.2f})\t"
              f"{orig_err:.2f}\t{corr_err:.2f}")

if __name__ == "__main__":
    test_correction_logic()