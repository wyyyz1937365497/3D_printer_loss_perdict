import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from train_model import DisplacementPredictor
from sklearn.metrics import r2_score

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

def main():
    print("验证训练好的模型效果...")
    
    # 加载数据
    data_path = '../data/printer_displacement_data.csv'
    if not os.path.exists(data_path):
        possible_paths = [
            '../data/printer_displacement_data.csv',
            'data/printer_displacement_data.csv',
            './data/printer_displacement_data.csv',
            'f:/AI/3D_printer_loss_perdict/data/printer_displacement_data.csv'
        ]
        
        found_path = None
        for path in possible_paths:
            if os.path.exists(path):
                found_path = path
                break
        
        if found_path is None:
            raise FileNotFoundError(f"数据文件不存在于任何已知路径: {possible_paths}")
        
        data_path = found_path
    
    df = pd.read_csv(data_path)
    print(f"加载数据: {df.shape[0]} 个样本")
    
    # 分离特征和标签
    feature_columns = ['velocity', 'radius', 'weight', 'stiffness', 'ideal_x', 'ideal_y']
    label_columns = ['displacement_x', 'displacement_y']
    
    X = df[feature_columns].values
    y_true = df[label_columns].values
    
    # 加载训练好的模型和标准化器
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
    
    # 标准化输入
    X_scaled = scaler_X.transform(X)
    
    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X_scaled)
    
    # 预测
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).numpy()
    
    # 反标准化预测结果
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # 计算R²分数
    r2_x = r2_score(y_true[:, 0], y_pred[:, 0])
    r2_y = r2_score(y_true[:, 1], y_pred[:, 1])
    
    print(f"X方向R²分数: {r2_x:.4f}")
    print(f"Y方向R²分数: {r2_y:.4f}")
    
    # 计算预测误差
    pred_errors = np.sqrt((y_true[:, 0] - y_pred[:, 0])**2 + (y_true[:, 1] - y_pred[:, 1])**2)
    print(f"平均预测误差: {np.mean(pred_errors):.4f}")
    print(f"最大预测误差: {np.max(pred_errors):.4f}")
    
    # 可视化预测结果
    plt.figure(figsize=(15, 5))
    
    # 子图1: 真实vs预测 X方向
    plt.subplot(1, 3, 1)
    plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5)
    plt.plot([y_true[:, 0].min(), y_true[:, 0].max()], 
             [y_true[:, 0].min(), y_true[:, 0].max()], 'r--', lw=2)
    plt.xlabel('真实X偏差')
    plt.ylabel('预测X偏差')
    plt.title(f'X方向预测效果\nR² = {r2_x:.4f}')
    plt.grid(True, alpha=0.3)
    
    # 子图2: 真实vs预测 Y方向
    plt.subplot(1, 3, 2)
    plt.scatter(y_true[:, 1], y_pred[:, 1], alpha=0.5)
    plt.plot([y_true[:, 1].min(), y_true[:, 1].max()], 
             [y_true[:, 1].min(), y_true[:, 1].max()], 'r--', lw=2)
    plt.xlabel('真实Y偏差')
    plt.ylabel('预测Y偏差')
    plt.title(f'Y方向预测效果\nR² = {r2_y:.4f}')
    plt.grid(True, alpha=0.3)
    
    # 子图3: 预测误差分布
    plt.subplot(1, 3, 3)
    plt.hist(pred_errors, bins=50, alpha=0.7, color='blue', density=True)
    plt.xlabel('预测误差 (mm)')
    plt.ylabel('概率密度')
    plt.title(f'预测误差分布\n平均误差: {np.mean(pred_errors):.4f}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(models_dir, 'model_validation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"验证结果已保存至: {output_path}")
    
    # 评估模型在尖锐转角和圆角上的表现
    radii = df['radius'].values
    sharp_corners = radii <= 5  # 尖锐转角
    rounded_corners = radii > 5  # 圆角
    
    if np.any(sharp_corners):
        sharp_errors = np.sqrt((y_true[sharp_corners, 0] - y_pred[sharp_corners, 0])**2 + 
                               (y_true[sharp_corners, 1] - y_pred[sharp_corners, 1])**2)
        print(f"尖锐转角平均预测误差: {np.mean(sharp_errors):.4f}")
    else:
        print("数据中没有尖锐转角样本")
    
    if np.any(rounded_corners):
        rounded_errors = np.sqrt((y_true[rounded_corners, 0] - y_pred[rounded_corners, 0])**2 + 
                                 (y_true[rounded_corners, 1] - y_pred[rounded_corners, 1])**2)
        print(f"圆角平均预测误差: {np.mean(rounded_errors):.4f}")
    else:
        print("数据中没有圆角样本")
    
    plt.show()
    
    print("模型验证完成！")

if __name__ == "__main__":
    main()