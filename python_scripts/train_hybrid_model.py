import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib
import os
import matplotlib.pyplot as plt
from hybrid_model import HybridDisplacementPredictor

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_process_sequence_data_with_velocity_and_next_pos(file_path):
    """
    加载并处理包含速度方向和下一点位置的序列数据
    """
    print("加载序列数据...")
    df = pd.read_csv(file_path)
    
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 提取基础特征
    feature_cols = ['velocity', 'radius', 'weight', 'stiffness']
    label_cols = ['displacement_x', 'displacement_y']
    
    # 重塑为序列格式
    sequence_length = 10  # 序列长度
    n_samples = len(df) // sequence_length
    
    X = df[feature_cols].values[:n_samples * sequence_length]
    y = df[label_cols].values[:n_samples * sequence_length]
    
    # 重塑为序列格式
    X = X.reshape(n_samples, sequence_length, -1)
    y = y.reshape(n_samples, sequence_length, -1)
    
    # 计算速度方向（当前点到下一点的方向向量）
    velocities = np.zeros_like(X[:, :, :2])  # 只取前两个维度作为速度方向
    
    # 对于每个序列，计算相邻点之间的方向
    for seq_idx in range(n_samples):
        for i in range(sequence_length - 1):
            # 使用理想位置计算方向（如果可用）或使用序列中的下一个点
            velocities[seq_idx, i, :] = [df.iloc[seq_idx * sequence_length + i + 1]['velocity'], 0]  # 这里简化处理
        
        # 最后一个点使用前一个点的方向
        if sequence_length > 1:
            velocities[seq_idx, -1, :] = velocities[seq_idx, -2, :]
        else:
            velocities[seq_idx, -1, :] = [0, 0]
    
    # 计算下一点位置（序列中的下一个点）
    next_positions = np.zeros_like(X[:, :, -2:])  # 取后两个维度作为位置（这里使用当前点位置作为近似）
    
    for seq_idx in range(n_samples):
        for i in range(sequence_length - 1):
            # 下一点位置（从原始数据中获取）
            next_positions[seq_idx, i, 0] = df.iloc[seq_idx * sequence_length + i + 1]['velocity']  # 简化处理
            next_positions[seq_idx, i, 1] = df.iloc[seq_idx * sequence_length + i + 1]['radius']    # 简化处理
        
        # 最后一个点没有"下一点"，使用当前点
        next_positions[seq_idx, -1, :] = [X[seq_idx, -1, 0], X[seq_idx, -1, 1]]
    
    print(f"基础特征矩阵形状: {X.shape}")
    print(f"标签矩阵形状: {y.shape}")
    print(f"速度方向矩阵形状: {velocities.shape}")
    print(f"下一点位置矩阵形状: {next_positions.shape}")
    
    return X, y, velocities, next_positions

def train_hybrid_model():
    """
    训练混合模型
    """
    print("开始训练混合神经网络模型...")
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'printer_sequence_displacement_data.csv')
    X, y, velocities, next_positions = load_and_process_sequence_data_with_velocity_and_next_pos(data_path)
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test, vel_train, vel_test, next_pos_train, next_pos_test = train_test_split(
        X, y, velocities, next_positions, test_size=0.2, random_state=42
    )
    
    # 标准化数据
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    vel_scaler = StandardScaler()
    next_pos_scaler = StandardScaler()
    
    # 重塑数据进行标准化
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    y_train_reshaped = y_train.reshape(-1, y_train.shape[-1])
    y_test_reshaped = y_test.reshape(-1, y_test.shape[-1])
    vel_train_reshaped = vel_train.reshape(-1, vel_train.shape[-1])
    vel_test_reshaped = vel_test.reshape(-1, vel_test.shape[-1])
    next_pos_train_reshaped = next_pos_train.reshape(-1, next_pos_train.shape[-1])
    next_pos_test_reshaped = next_pos_test.reshape(-1, next_pos_test.shape[-1])
    
    X_train_scaled = X_scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_test_scaled = X_scaler.transform(X_test_reshaped).reshape(X_test.shape)
    y_train_scaled = y_scaler.fit_transform(y_train_reshaped).reshape(y_train.shape)
    y_test_scaled = y_scaler.transform(y_test_reshaped).reshape(y_test.shape)
    vel_train_scaled = vel_scaler.fit_transform(vel_train_reshaped).reshape(vel_train.shape)
    vel_test_scaled = vel_scaler.transform(vel_test_reshaped).reshape(vel_test.shape)
    next_pos_train_scaled = next_pos_scaler.fit_transform(next_pos_train_reshaped).reshape(next_pos_train.shape)
    next_pos_test_scaled = next_pos_scaler.transform(next_pos_test_reshaped).reshape(next_pos_test.shape)
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_test_tensor = torch.FloatTensor(y_test_scaled).to(device)
    vel_train_tensor = torch.FloatTensor(vel_train_scaled).to(device)
    vel_test_tensor = torch.FloatTensor(vel_test_scaled).to(device)
    next_pos_train_tensor = torch.FloatTensor(next_pos_train_scaled).to(device)
    next_pos_test_tensor = torch.FloatTensor(next_pos_test_scaled).to(device)
    
    print(f"训练集形状 - X: {X_train_tensor.shape}, y: {y_train_tensor.shape}")
    print(f"测试集形状 - X: {X_test_tensor.shape}, y: {y_test_tensor.shape}")
    
    # 初始化混合模型
    model = HybridDisplacementPredictor(
        input_size=4,  # velocity, radius, weight, stiffness
        d_model=128,
        nhead=8,
        num_layers=2,
        output_size=2,  # displacement_x, displacement_y
        sequence_length=10,
        dropout=0.1
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
    
    # 训练参数
    num_epochs = 300
    batch_size = 32
    train_losses = []
    val_losses = []
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor, vel_train_tensor, next_pos_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print("开始训练...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y, batch_vel, batch_next_pos in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X, batch_vel, batch_next_pos)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # 计算平均训练损失
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor, vel_test_tensor, next_pos_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor)
            val_losses.append(val_loss.item())
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        scheduler.step()
    
    print("训练完成!")
    
    # 保存模型和标准化器
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(project_dir, 'models')
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    model_path = os.path.join(models_dir, 'hybrid_displacement_predictor.pth')
    torch.save(model.state_dict(), model_path)
    
    scaler_x_path = os.path.join(models_dir, 'hybrid_scaler_X.pkl')
    scaler_y_path = os.path.join(models_dir, 'hybrid_scaler_y.pkl')
    scaler_vel_path = os.path.join(models_dir, 'hybrid_scaler_vel.pkl')
    scaler_next_pos_path = os.path.join(models_dir, 'hybrid_scaler_next_pos.pkl')
    
    joblib.dump(X_scaler, scaler_x_path)
    joblib.dump(y_scaler, scaler_y_path)
    joblib.dump(vel_scaler, scaler_vel_path)
    joblib.dump(next_pos_scaler, scaler_next_pos_path)
    
    print(f"模型保存至: {model_path}")
    print(f"标准化器保存至: {scaler_x_path}, {scaler_y_path}, {scaler_vel_path}, {scaler_next_pos_path}")
    
    # 反标准化预测结果进行评估
    model.eval()
    with torch.no_grad():
        test_predictions_scaled = model(X_test_tensor, vel_test_tensor, next_pos_test_tensor)
        test_predictions = y_scaler.inverse_transform(test_predictions_scaled.cpu().numpy().reshape(-1, 2))
        y_test_flat = y_scaler.inverse_transform(y_test_tensor.cpu().numpy().reshape(-1, 2))
    
    # 计算R²分数
    r2_x = r2_score(y_test_flat[:, 0], test_predictions[:, 0])
    r2_y = r2_score(y_test_flat[:, 1], test_predictions[:, 1])
    
    print(f"X方向R2分数: {r2_x:.4f}")
    print(f"Y方向R2分数: {r2_y:.4f}")
    
    # 可视化训练结果
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('训练过程')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test_flat[:, 0], test_predictions[:, 0], alpha=0.5, label='X方向')
    plt.scatter(y_test_flat[:, 1], test_predictions[:, 1], alpha=0.5, label='Y方向')
    plt.plot([y_test_flat.min(), y_test_flat.max()], [y_test_flat.min(), y_test_flat.max()], 'r--', lw=2)
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('预测值 vs 真实值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存可视化结果
    vis_path = os.path.join(models_dir, 'hybrid_training_results.png')
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    print(f"训练结果可视化已保存至: {vis_path}")
    
    return model, X_scaler, y_scaler, vel_scaler, next_pos_scaler

if __name__ == "__main__":
    model, X_scaler, y_scaler, vel_scaler, next_pos_scaler = train_hybrid_model()