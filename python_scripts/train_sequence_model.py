import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import matplotlib.pyplot as plt
from sequence_model import SequenceDisplacementPredictor

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_process_sequence_data(file_path):
    """
    加载并处理序列数据
    """
    print("加载序列数据...")
    df = pd.read_csv(file_path)
    
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 提取特征和标签
    feature_cols = ['velocity', 'radius', 'weight', 'stiffness']
    label_cols = ['displacement_x', 'displacement_y']
    
    X = df[feature_cols].values
    y = df[label_cols].values
    
    # 重塑为序列格式
    sequence_length = 10  # 序列长度
    n_samples = len(df) // sequence_length
    X = X[:n_samples * sequence_length].reshape(n_samples, sequence_length, -1)
    y = y[:n_samples * sequence_length].reshape(n_samples, sequence_length, -1)
    
    print(f"特征矩阵形状: {X.shape}")
    print(f"标签矩阵形状: {y.shape}")
    
    return X, y

def train_sequence_model():
    """
    训练序列模型
    """
    print("开始训练序列神经网络模型...")
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'printer_sequence_displacement_data.csv')
    X, y = load_and_process_sequence_data(data_path)
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化数据
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    # 重塑数据进行标准化
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    y_train_reshaped = y_train.reshape(-1, y_train.shape[-1])
    y_test_reshaped = y_test.reshape(-1, y_test.shape[-1])
    
    X_train_scaled = X_scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_test_scaled = X_scaler.transform(X_test_reshaped).reshape(X_test.shape)
    y_train_scaled = y_scaler.fit_transform(y_train_reshaped).reshape(y_train.shape)
    y_test_scaled = y_scaler.transform(y_test_reshaped).reshape(y_test.shape)
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_test_tensor = torch.FloatTensor(y_test_scaled).to(device)
    
    print(f"训练集形状 - X: {X_train_tensor.shape}, y: {y_train_tensor.shape}")
    print(f"测试集形状 - X: {X_test_tensor.shape}, y: {y_test_tensor.shape}")
    
    # 初始化模型
    model = SequenceDisplacementPredictor(
        input_size=4,  # velocity, radius, weight, stiffness
        hidden_size=128,
        num_layers=2,
        output_size=2,  # displacement_x, displacement_y
        sequence_length=10
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # 训练参数
    num_epochs = 300
    batch_size = 32
    train_losses = []
    val_losses = []
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print("开始训练...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
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
            val_outputs = model(X_test_tensor)
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
    
    model_path = os.path.join(models_dir, 'sequence_displacement_predictor.pth')
    torch.save(model.state_dict(), model_path)
    
    scaler_x_path = os.path.join(models_dir, 'sequence_scaler_X.pkl')
    scaler_y_path = os.path.join(models_dir, 'sequence_scaler_y.pkl')
    joblib.dump(X_scaler, scaler_x_path)
    joblib.dump(y_scaler, scaler_y_path)
    
    print(f"模型保存至: {model_path}")
    print(f"标准化器保存至: {scaler_x_path}, {scaler_y_path}")
    
    # 反标准化预测结果进行评估
    model.eval()
    with torch.no_grad():
        test_predictions_scaled = model(X_test_tensor)
        test_predictions = y_scaler.inverse_transform(test_predictions_scaled.cpu().numpy().reshape(-1, 2))
        y_test_flat = y_scaler.inverse_transform(y_test_tensor.cpu().numpy().reshape(-1, 2))
    
    # 计算R²分数
    from sklearn.metrics import r2_score
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
    vis_path = os.path.join(models_dir, 'sequence_training_results.png')
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    print(f"训练结果可视化已保存至: {vis_path}")
    
    return model, X_scaler, y_scaler

if __name__ == "__main__":
    model, X_scaler, y_scaler = train_sequence_model()