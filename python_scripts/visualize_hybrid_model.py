import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
from hybrid_model import HybridDisplacementPredictor

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

def load_and_visualize_hybrid_model():
    """
    加载并可视化混合模型的性能
    """
    print("加载混合模型并进行可视化...")
    
    # 加载模型和标准化器
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(project_dir, 'models')
    
    model_path = os.path.join(models_dir, 'hybrid_displacement_predictor.pth')
    scaler_x_path = os.path.join(models_dir, 'hybrid_scaler_X.pkl')
    scaler_y_path = os.path.join(models_dir, 'hybrid_scaler_y.pkl')
    scaler_vel_path = os.path.join(models_dir, 'hybrid_scaler_vel.pkl')
    scaler_next_pos_path = os.path.join(models_dir, 'hybrid_scaler_next_pos.pkl')
    
    # 初始化并加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridDisplacementPredictor(
        input_size=4, 
        d_model=128, 
        nhead=8, 
        num_layers=2, 
        output_size=2, 
        sequence_length=10
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 加载标准化器
    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    scaler_vel = joblib.load(scaler_vel_path)
    scaler_next_pos = joblib.load(scaler_next_pos_path)
    
    # 加载测试数据
    data_path = os.path.join(project_dir, 'data', 'printer_sequence_displacement_data.csv')
    df = pd.read_csv(data_path)
    
    # 准备序列数据
    feature_cols = ['velocity', 'radius', 'weight', 'stiffness']
    label_cols = ['displacement_x', 'displacement_y']
    
    # 重塑为序列格式
    sequence_length = 10
    n_samples = len(df) // sequence_length
    
    X = df[feature_cols].values[:n_samples * sequence_length]
    y = df[label_cols].values[:n_samples * sequence_length]
    
    # 重塑为序列格式
    X = X.reshape(n_samples, sequence_length, -1)
    y = y.reshape(n_samples, sequence_length, -1)
    
    # 计算速度方向
    velocities = np.zeros_like(X[:, :, :2])
    for seq_idx in range(n_samples):
        for i in range(sequence_length - 1):
            # 计算相邻点之间的方向
            velocities[seq_idx, i, 0] = X[seq_idx, i+1, 0] - X[seq_idx, i, 0]  # 速度x分量
            velocities[seq_idx, i, 1] = X[seq_idx, i+1, 1] - X[seq_idx, i, 1]  # 速度y分量
        
        # 最后一个点使用前一个点的方向
        if sequence_length > 1:
            velocities[seq_idx, -1, :] = velocities[seq_idx, -2, :]
    
    # 计算下一点位置
    next_positions = np.zeros_like(X[:, :, -2:])
    for seq_idx in range(n_samples):
        for i in range(sequence_length - 1):
            next_positions[seq_idx, i, :] = [X[seq_idx, i+1, 0], X[seq_idx, i+1, 1]]
        
        # 最后一个点使用当前点
        next_positions[seq_idx, -1, :] = [X[seq_idx, -1, 0], X[seq_idx, -1, 1]]
    
    # 分割数据
    split_idx = int(n_samples * 0.8)
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    vel_test = velocities[split_idx:]
    next_pos_test = next_positions[split_idx:]
    
    # 标准化测试数据
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    y_test_reshaped = y_test.reshape(-1, y_test.shape[-1])
    vel_test_reshaped = vel_test.reshape(-1, vel_test.shape[-1])
    next_pos_test_reshaped = next_pos_test.reshape(-1, next_pos_test.shape[-1])
    
    X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)
    y_test_scaled = scaler_y.transform(y_test_reshaped).reshape(y_test.shape)
    vel_test_scaled = scaler_vel.transform(vel_test_reshaped).reshape(vel_test.shape)
    next_pos_test_scaled = scaler_next_pos.transform(next_pos_test_reshaped).reshape(next_pos_test.shape)
    
    # 转换为PyTorch张量
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_test_tensor = torch.FloatTensor(y_test_scaled).to(device)
    vel_test_tensor = torch.FloatTensor(vel_test_scaled).to(device)
    next_pos_test_tensor = torch.FloatTensor(next_pos_test_scaled).to(device)
    
    print(f"测试集形状 - X: {X_test_tensor.shape}, y: {y_test_tensor.shape}")
    
    # 进行预测
    with torch.no_grad():
        predictions_scaled = model(X_test_tensor, vel_test_tensor, next_pos_test_tensor)
        predictions_scaled = predictions_scaled.cpu().numpy()
    
    # 反标准化预测结果
    predictions_flat = predictions_scaled.reshape(-1, 2)
    y_test_flat = y_test_tensor.cpu().numpy().reshape(-1, 2)
    
    predictions = scaler_y.inverse_transform(predictions_flat)
    y_test_original = scaler_y.inverse_transform(y_test_flat)
    
    # 计算R²分数
    r2_x = r2_score(y_test_original[:, 0], predictions[:, 0])
    r2_y = r2_score(y_test_original[:, 1], predictions[:, 1])
    
    print(f"混合模型 - X方向R2分数: {r2_x:.4f}")
    print(f"混合模型 - Y方向R2分数: {r2_y:.4f}")
    
    # 加载传统模型进行对比
    from train_model import DisplacementPredictor
    traditional_model_path = os.path.join(models_dir, 'displacement_predictor.pth')
    traditional_scaler_x_path = os.path.join(models_dir, 'scaler_X.pkl')
    traditional_scaler_y_path = os.path.join(models_dir, 'scaler_y.pkl')
    
    traditional_model = DisplacementPredictor()
    traditional_model.load_state_dict(torch.load(traditional_model_path, map_location=device))
    traditional_model.to(device)
    traditional_model.eval()
    
    traditional_scaler_X = joblib.load(traditional_scaler_x_path)
    traditional_scaler_y = joblib.load(traditional_scaler_y_path)
    
    # 加载并准备传统模型的测试数据（非序列数据）
    original_data_path = os.path.join(project_dir, 'data', 'printer_displacement_data.csv')
    original_df = pd.read_csv(original_data_path)
    
    # 提取特征和标签
    feature_cols = ['velocity', 'radius', 'weight', 'stiffness', 'ideal_x', 'ideal_y']
    label_cols = ['displacement_x', 'displacement_y']
    
    X_original = original_df[feature_cols].values
    y_original = original_df[label_cols].values
    
    # 分割数据集
    split_idx = int(len(X_original) * 0.8)
    X_test_original = X_original[split_idx:]
    y_test_original_traditional = y_original[split_idx:]
    
    # 标准化传统模型的测试数据
    X_test_original_scaled = traditional_scaler_X.transform(X_test_original)
    y_test_original_scaled = traditional_scaler_y.transform(y_test_original_traditional)
    
    X_test_original_tensor = torch.FloatTensor(X_test_original_scaled).to(device)
    y_test_original_tensor = torch.FloatTensor(y_test_original_scaled).to(device)
    
    with torch.no_grad():
        traditional_predictions_scaled = traditional_model(X_test_original_tensor)
        traditional_predictions_scaled = traditional_predictions_scaled.cpu().numpy()
    
    traditional_predictions = traditional_scaler_y.inverse_transform(traditional_predictions_scaled)
    y_test_original_traditional = traditional_scaler_y.inverse_transform(y_test_original_scaled)
    
    # 计算传统模型R²分数
    traditional_r2_x = r2_score(y_test_original_traditional[:, 0], traditional_predictions[:, 0])
    traditional_r2_y = r2_score(y_test_original_traditional[:, 1], traditional_predictions[:, 1])
    
    print(f"传统模型 - X方向R2分数: {traditional_r2_x:.4f}")
    print(f"传统模型 - Y方向R2分数: {traditional_r2_y:.4f}")
    
    # 创建可视化图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 混合模型：X方向预测
    axes[0, 0].scatter(y_test_original[:, 0], predictions[:, 0], alpha=0.5)
    axes[0, 0].plot([y_test_original[:, 0].min(), y_test_original[:, 0].max()], 
                     [y_test_original[:, 0].min(), y_test_original[:, 0].max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('真实值')
    axes[0, 0].set_ylabel('预测值')
    axes[0, 0].set_title(f'混合模型 - X方向\nR² = {r2_x:.4f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 混合模型：Y方向预测
    axes[0, 1].scatter(y_test_original[:, 1], predictions[:, 1], alpha=0.5, color='orange')
    axes[0, 1].plot([y_test_original[:, 1].min(), y_test_original[:, 1].max()], 
                     [y_test_original[:, 1].min(), y_test_original[:, 1].max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('真实值')
    axes[0, 1].set_ylabel('预测值')
    axes[0, 1].set_title(f'混合模型 - Y方向\nR² = {r2_y:.4f}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 混合模型：误差分布
    x_errors = y_test_original[:, 0] - predictions[:, 0]
    y_errors = y_test_original[:, 1] - predictions[:, 1]
    axes[0, 2].hist([x_errors, y_errors], bins=50, alpha=0.7, label=['X方向误差', 'Y方向误差'], color=['blue', 'orange'])
    axes[0, 2].set_xlabel('误差')
    axes[0, 2].set_ylabel('频次')
    axes[0, 2].set_title('混合模型误差分布')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 传统模型：X方向预测
    axes[1, 0].scatter(y_test_original_traditional[:, 0], traditional_predictions[:, 0], alpha=0.5, color='green')
    axes[1, 0].plot([y_test_original_traditional[:, 0].min(), y_test_original_traditional[:, 0].max()], 
                     [y_test_original_traditional[:, 0].min(), y_test_original_traditional[:, 0].max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('真实值')
    axes[1, 0].set_ylabel('预测值')
    axes[1, 0].set_title(f'传统模型 - X方向\nR² = {traditional_r2_x:.4f}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 传统模型：Y方向预测
    axes[1, 1].scatter(y_test_original_traditional[:, 1], traditional_predictions[:, 1], alpha=0.5, color='red')
    axes[1, 1].plot([y_test_original_traditional[:, 1].min(), y_test_original_traditional[:, 1].max()], 
                     [y_test_original_traditional[:, 1].min(), y_test_original_traditional[:, 1].max()], 'r--', lw=2)
    axes[1, 1].set_xlabel('真实值')
    axes[1, 1].set_ylabel('预测值')
    axes[1, 1].set_title(f'传统模型 - Y方向\nR² = {traditional_r2_y:.4f}')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 传统模型：误差分布
    x_traditional_errors = y_test_original_traditional[:, 0] - traditional_predictions[:, 0]
    y_traditional_errors = y_test_original_traditional[:, 1] - traditional_predictions[:, 1]
    axes[1, 2].hist([x_traditional_errors, y_traditional_errors], bins=50, alpha=0.7, 
                     label=['X方向误差', 'Y方向误差'], color=['green', 'red'])
    axes[1, 2].set_xlabel('误差')
    axes[1, 2].set_ylabel('频次')
    axes[1, 2].set_title('传统模型误差分布')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存可视化结果
    vis_path = os.path.join(models_dir, 'hybrid_model_comparison.png')
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    print(f"混合模型可视化结果已保存至: {vis_path}")
    
    # 创建序列预测可视化
    plt.figure(figsize=(15, 10))
    
    # 选择几个序列进行可视化
    n_sequences_to_show = 3
    for i in range(n_sequences_to_show):
        plt.subplot(n_sequences_to_show, 2, 2*i+1)
        plt.plot(y_test[i, :, 0], label='真实X方向偏差', marker='o')
        plt.plot(predictions_scaled[i, :, 0], label='预测X方向偏差', marker='s')
        plt.title(f'序列 {i+1} - X方向偏差')
        plt.xlabel('序列位置')
        plt.ylabel('偏差')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(n_sequences_to_show, 2, 2*i+2)
        plt.plot(y_test[i, :, 1], label='真实Y方向偏差', marker='o', color='orange')
        plt.plot(predictions_scaled[i, :, 1], label='预测Y方向偏差', marker='s', color='red')
        plt.title(f'序列 {i+1} - Y方向偏差')
        plt.xlabel('序列位置')
        plt.ylabel('偏差')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存序列可视化结果
    seq_vis_path = os.path.join(models_dir, 'hybrid_model_sequence_predictions.png')
    plt.savefig(seq_vis_path, dpi=300, bbox_inches='tight')
    print(f"混合模型序列预测可视化已保存至: {seq_vis_path}")
    
    plt.show()
    
    print("可视化完成!")

if __name__ == "__main__":
    load_and_visualize_hybrid_model()