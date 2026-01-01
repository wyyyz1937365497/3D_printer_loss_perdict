import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from data_processing import load_and_process_data
import os
import matplotlib.pyplot as plt

# 定义神经网络模型
class DisplacementPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_sizes=[128, 256, 128], output_size=2):
        super(DisplacementPredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # 添加dropout防止过拟合
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def train_model():
    print("开始训练神经网络模型...")
    
    # 加载数据
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = load_and_process_data()
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 初始化模型
    model = DisplacementPredictor()
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # 训练参数
    num_epochs = 300
    train_losses = []
    val_losses = []
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证损失
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor)
        
        train_loss /= len(train_loader)
        val_loss = val_loss.item()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.6f}, '
                  f'Val Loss: {val_loss:.6f}')
    
    print("训练完成!")
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 预测结果可视化
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor).numpy()
        
        # 反标准化
        test_predictions_orig = scaler_y.inverse_transform(test_predictions)
        y_test_orig = scaler_y.inverse_transform(y_test_tensor.numpy())
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test_orig[:, 0], test_predictions_orig[:, 0], alpha=0.5, label='X direction')
    plt.plot([y_test_orig[:, 0].min(), y_test_orig[:, 0].max()], 
             [y_test_orig[:, 0].min(), y_test_orig[:, 0].max()], 'r--', lw=2)
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.title('X Direction Displacement Prediction')
    plt.legend()
    
    plt.tight_layout()
    
    # 确保模型目录存在并使用绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(project_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 保存图片到正确的模型目录
    plot_path = os.path.join(models_dir, 'training_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以避免内存问题
    
    # 计算R²分数
    from sklearn.metrics import r2_score
    r2_x = r2_score(y_test_orig[:, 0], test_predictions_orig[:, 0])
    r2_y = r2_score(y_test_orig[:, 1], test_predictions_orig[:, 1])
    print(f"X direction R2 score: {r2_x:.4f}")
    print(f"Y direction R2 score: {r2_y:.4f}")
    
    # 使用绝对路径保存模型
    model_path = os.path.join(models_dir, 'displacement_predictor.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    return model, scaler_X, scaler_y

if __name__ == "__main__":
    # 确保模型目录存在
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(project_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 训练模型
    model, scaler_X, scaler_y = train_model()
    
    # 使用绝对路径保存scalers
    import joblib
    scaler_x_path = os.path.join(models_dir, 'scaler_X.pkl')
    scaler_y_path = os.path.join(models_dir, 'scaler_y.pkl')
    joblib.dump(scaler_X, scaler_x_path)
    joblib.dump(scaler_y, scaler_y_path)
    print("Scalers saved")