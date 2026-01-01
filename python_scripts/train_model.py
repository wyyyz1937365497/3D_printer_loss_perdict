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

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

class DisplacementPredictor(nn.Module):
    """
    位移预测模型
    """
    def __init__(self, input_size=6, hidden_sizes=[128, 256, 128], output_size=2):
        super(DisplacementPredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def train_model():
    """
    训练模型的主要函数
    """
    print("开始训练神经网络模型...")
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    data_path = os.path.join(data_dir, 'printer_displacement_data.csv')
    
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 提取特征和标签
    feature_cols = ['velocity', 'radius', 'weight', 'stiffness', 'ideal_x', 'ideal_y']
    label_cols = ['displacement_x', 'displacement_y']
    
    X = df[feature_cols].values
    y = df[label_cols].values
    
    print(f"特征矩阵形状: {X.shape}")
    print(f"标签矩阵形状: {y.shape}")
    
    # 标准化数据
    print("标准化数据...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # 分割训练集和测试集
    print("分割训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    print(f"训练集形状 - X: {X_train.shape}, y: {y_train.shape}")
    print(f"测试集形状 - X: {X_test.shape}, y: {y_test.shape}")
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # 初始化模型
    model = DisplacementPredictor().to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    # 训练参数
    num_epochs = 300
    batch_size = 64
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
    
    model_path = os.path.join(models_dir, 'displacement_predictor.pth')
    torch.save(model.state_dict(), model_path)
    
    scaler_x_path = os.path.join(models_dir, 'scaler_X.pkl')
    scaler_y_path = os.path.join(models_dir, 'scaler_y.pkl')
    joblib.dump(scaler_X, scaler_x_path)
    joblib.dump(scaler_y, scaler_y_path)
    
    print(f"模型保存至: {model_path}")
    print(f"标准化器保存至: {scaler_x_path}, {scaler_y_path}")
    
    # 反标准化预测结果进行评估
    model.eval()
    with torch.no_grad():
        test_predictions_scaled = model(X_test_tensor)
        test_predictions = scaler_y.inverse_transform(test_predictions_scaled.cpu().numpy())
        y_test_original = scaler_y.inverse_transform(y_test_tensor.cpu().numpy())
    
    # 计算R²分数
    r2_x = r2_score(y_test_original[:, 0], test_predictions[:, 0])
    r2_y = r2_score(y_test_original[:, 1], test_predictions[:, 1])
    
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
    plt.scatter(y_test_original[:, 0], test_predictions[:, 0], alpha=0.5, label='X方向')
    plt.scatter(y_test_original[:, 1], test_predictions[:, 1], alpha=0.5, label='Y方向')
    plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('预测值 vs 真实值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存可视化结果
    vis_path = os.path.join(models_dir, 'training_results.png')
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    print(f"训练结果可视化已保存至: {vis_path}")
    
    return model, scaler_X, scaler_y

if __name__ == "__main__":
    train_model()