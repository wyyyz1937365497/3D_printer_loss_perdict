import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from train_model import DisplacementPredictor

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

def validate_correction_with_real_data():
    print("使用真实数据验证校正效果...")
    
    # 加载数据集
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'printer_displacement_data.csv')
    df = pd.read_csv(data_path)
    
    # 随机选择一些样本进行测试
    sample_size = 50
    sample_indices = np.random.choice(df.index, size=sample_size, replace=False)
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
    
    # 计算改善百分比
    improvements = ((original_errors - corrected_errors) / original_errors) * 100
    
    print(f"改善样本比例: {np.sum(improvements > 0) / len(improvements) * 100:.2f}%")
    print(f"平均改善百分比: {np.mean(improvements):.2f}%")
    
    # 可视化结果
    plt.figure(figsize=(15, 5))
    
    # 子图1: 原始vs校正误差对比
    plt.subplot(1, 3, 1)
    plt.scatter(original_errors, corrected_errors, alpha=0.6)
    plt.plot([0, max(original_errors)], [0, max(original_errors)], 'r--', lw=2, label='y=x (无改善线)')
    plt.xlabel('原始误差 (mm)')
    plt.ylabel('校正后误差 (mm)')
    plt.title('原始误差 vs 校正后误差')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 误差改善分布
    plt.subplot(1, 3, 2)
    plt.hist(improvements, bins=30, alpha=0.7, color='green', density=True)
    plt.xlabel('改善百分比 (%)')
    plt.ylabel('频率密度')
    plt.title(f'误差改善分布\n平均改善: {np.mean(improvements):.2f}%')
    plt.grid(True, alpha=0.3)
    
    # 子图3: 路径对比（前20个点）
    plt.subplot(1, 3, 3)
    n_show = min(20, len(ideal_x))
    plt.plot(ideal_x[:n_show], ideal_y[:n_show], 'g-', linewidth=2, label='理想路径', alpha=0.8, marker='o', markersize=6)
    plt.plot(actual_x[:n_show], actual_y[:n_show], 'r-', linewidth=1, label='有误差路径', alpha=0.6, marker='s', markersize=4)
    plt.plot(corrected_x[:n_show], corrected_y[:n_show], 'b-', linewidth=1, label='校正后路径', alpha=0.6, marker='^', markersize=4)
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.title('路径对比 (前20个点)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(models_dir, 'real_data_correction_validation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"验证结果已保存至: {output_path}")
    
    print("\n详细比较（前5个样本）:")
    print("样本\t理想位置\t\t实际位置\t\t预测偏差\t\t校正后位置\t\t原始误差\t校正误差")
    for i in range(min(5, len(sample_data))):
        orig_err = np.sqrt((actual_x[i] - ideal_x[i])**2 + (actual_y[i] - ideal_y[i])**2)
        corr_err = np.sqrt((corrected_x[i] - ideal_x[i])**2 + (corrected_y[i] - ideal_y[i])**2)
        improvement = ((orig_err - corr_err) / orig_err * 100)
        print(f"{i}\t({ideal_x[i]:.2f}, {ideal_y[i]:.2f})\t({actual_x[i]:.2f}, {actual_y[i]:.2f})\t"
              f"({y_pred[i, 0]:.2f}, {y_pred[i, 1]:.2f})\t({corrected_x[i]:.2f}, {corrected_y[i]:.2f})\t"
              f"{orig_err:.2f}\t{corr_err:.2f}\t{improvement:+.1f}%")

if __name__ == "__main__":
    validate_correction_with_real_data()