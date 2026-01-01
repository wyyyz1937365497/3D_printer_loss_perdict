import numpy as np
import matplotlib.pyplot as plt
import torch
import joblib
import os
from train_model import DisplacementPredictor

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

def generate_ideal_path():
    """
    生成理想打印路径，包括直线段和转角段
    """
    # 创建一个包含直线和转角的路径
    # 路径：从原点出发，画一个矩形，然后画一个星形
    path_segments = []
    
    # 矩形路径
    rect_points = [
        (0, 0), (20, 0), (20, 20), (0, 20), (0, 0)  # 矩形
    ]
    
    # 星形路径（简化版）
    star_center = (30, 10)
    star_radius = 8
    star_points = []
    for i in range(5):
        angle = i * 2 * np.pi / 5
        # 外部点
        x_outer = star_center[0] + star_radius * np.cos(angle)
        y_outer = star_center[1] + star_radius * np.sin(angle)
        star_points.append((x_outer, y_outer))
        
        # 内部点
        angle_inner = angle + np.pi / 5
        x_inner = star_center[0] + star_radius * 0.5 * np.cos(angle_inner)
        y_inner = star_center[1] + star_radius * 0.5 * np.sin(angle_inner)
        star_points.append((x_inner, y_inner))
    
    # 将星形闭合
    star_points.append(star_points[0])
    
    ideal_path = rect_points + star_points
    
    return ideal_path

def simulate_physical_errors(ideal_path, velocities):
    """
    模拟物理因素导致的打印误差
    """
    noisy_path = []
    
    for i, (x, y) in enumerate(ideal_path):
        if i < len(velocities):
            v = velocities[i]
        else:
            v = velocities[-1]  # 如果速度不够，使用最后一个速度
            
        # 计算转角半径（简化计算）
        if i > 0 and i < len(ideal_path) - 1:
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
        
        # 模拟物理误差：速度越快、半径越小（转角越尖锐），误差越大
        error_factor = (v**2) / (radius * 10)  # 简化的物理模型
        
        # 添加随机误差和系统误差
        random_error_x = np.random.normal(0, error_factor)
        random_error_y = np.random.normal(0, error_factor)
        
        # 系统误差（与转角相关的偏差）
        if radius < 10:  # 尖锐转角
            sys_error_x = np.random.normal(0, error_factor * 1.5)  # 尖锐转角误差更大
            sys_error_y = np.random.normal(0, error_factor * 1.5)
        else:
            sys_error_x = np.random.normal(0, error_factor)
            sys_error_y = np.random.normal(0, error_factor)
        
        # 总误差
        total_error_x = random_error_x + sys_error_x
        total_error_y = random_error_y + sys_error_y
        
        # 应用误差到理想位置
        noisy_x = x + total_error_x
        noisy_y = y + total_error_y
        
        noisy_path.append((noisy_x, noisy_y))
    
    return noisy_path

def apply_model_correction(noisy_path, velocities, model, scaler_X, scaler_y):
    """
    应用训练好的模型对有误差的路径进行修正
    注意：这里是对noisy_path进行修正，而不是对ideal_path
    """
    corrected_path = []
    
    for i, (x, y) in enumerate(noisy_path):
        if i < len(velocities):
            v = velocities[i]
        else:
            v = velocities[-1]  # 如果速度不够，使用最后一个速度
            
        # 计算转角半径（基于理想路径计算，因为这是实际打印时的路径形状）
        ideal_path = generate_ideal_path()
        if i > 0 and i < len(ideal_path) - 1:
            prev_x, prev_y = ideal_path[i-1]
            next_x, next_y = ideal_path[i+1]
            
            # 使用三点法估算曲率半径
            a = np.sqrt((ideal_path[i][0] - prev_x)**2 + (ideal_path[i][1] - prev_y)**2)
            b = np.sqrt((next_x - ideal_path[i][0])**2 + (next_y - ideal_path[i][1])**2)
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
        
        # 固定参数
        weight = 10  # 喷头重量
        stiffness = 1.0  # 支撑梁刚度
        
        # 准备输入数据 - 使用有误差的位置作为输入
        input_data = np.array([[v, radius, weight, stiffness, x, y]])
        
        # 标准化输入
        input_scaled = scaler_X.transform(input_data)
        
        # 转换为PyTorch张量
        input_tensor = torch.FloatTensor(input_scaled)
        
        # 预测偏差
        with torch.no_grad():
            prediction_scaled = model(input_tensor)
            prediction_scaled = prediction_scaled.numpy()
        
        # 反标准化预测结果
        prediction = scaler_y.inverse_transform(prediction_scaled)
        
        # 应用校正（将预测的偏差加到有误差的位置上，以抵消原始偏差）
        corrected_x = x + prediction[0][0]
        corrected_y = y + prediction[0][1]
        
        corrected_path.append((corrected_x, corrected_y))
    
    return corrected_path

def calculate_errors(ideal_path, actual_path):
    """
    计算路径误差
    """
    errors = []
    for (ix, iy), (ax, ay) in zip(ideal_path, actual_path):
        error = np.sqrt((ix - ax)**2 + (iy - ay)**2)
        errors.append(error)
    return np.array(errors)

def main():
    print("开始可视化3D打印质量改进效果...")
    
    # 生成理想路径
    ideal_path = generate_ideal_path()
    print(f"生成了 {len(ideal_path)} 个路径点")
    
    # 定义速度（模拟不同段的不同速度）
    velocities = [50] * 5 + [80] * 10 + [60] * 5 + [100] * 11  # 匹配路径点数量
    
    # 模拟物理误差
    noisy_path = simulate_physical_errors(ideal_path, velocities)
    
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
    
    # 应用模型校正（对有误差的路径进行校正）
    corrected_path = apply_model_correction(noisy_path, velocities, model, scaler_X, scaler_y)
    
    # 计算误差
    noisy_errors = calculate_errors(ideal_path, noisy_path)
    corrected_errors = calculate_errors(ideal_path, corrected_path)
    
    # 打印统计信息
    print(f"平均误差 - 理想路径: 0.0000")
    print(f"平均误差 - 有噪声路径: {np.mean(noisy_errors):.4f}")
    print(f"平均误差 - 校正后路径: {np.mean(corrected_errors):.4f}")
    improvement = ((np.mean(noisy_errors) - np.mean(corrected_errors)) / np.mean(noisy_errors) * 100)
    print(f"误差减少: {improvement:.2f}%")
    
    # 创建可视化图表
    plt.figure(figsize=(18, 6))
    
    # 子图1: 路径对比
    plt.subplot(1, 3, 1)
    ideal_x, ideal_y = zip(*ideal_path)
    noisy_x, noisy_y = zip(*noisy_path)
    corrected_x, corrected_y = zip(*corrected_path)
    
    plt.plot(ideal_x, ideal_y, 'g-', linewidth=2, label='理想路径', alpha=0.8)
    plt.plot(noisy_x, noisy_y, 'r-', linewidth=1, label='有误差路径', alpha=0.6)
    plt.plot(corrected_x, corrected_y, 'b-', linewidth=1, label='校正后路径', alpha=0.6)
    
    # 标记起点
    plt.plot(ideal_x[0], ideal_y[0], 'go', markersize=8, label='起点')
    plt.plot(noisy_x[0], noisy_y[0], 'ro', markersize=6)
    plt.plot(corrected_x[0], corrected_y[0], 'bo', markersize=6)
    
    plt.title('3D打印路径对比')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 子图2: 误差对比
    plt.subplot(1, 3, 2)
    plt.plot(noisy_errors, 'r-', linewidth=2, label='有误差路径误差', alpha=0.8)
    plt.plot(corrected_errors, 'b-', linewidth=2, label='校正后路径误差', alpha=0.8)
    plt.title('路径点误差对比')
    plt.xlabel('路径点索引')
    plt.ylabel('误差 (mm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3: 误差分布直方图
    plt.subplot(1, 3, 3)
    plt.hist(noisy_errors, bins=30, alpha=0.6, label='有误差路径', color='red', density=True)
    plt.hist(corrected_errors, bins=30, alpha=0.6, label='校正后路径', color='blue', density=True)
    plt.title('误差分布对比')
    plt.xlabel('误差 (mm)')
    plt.ylabel('概率密度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(models_dir, 'quality_improvement_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存至: {output_path}")
    
    # 显示统计摘要
    print("\n=== 打印质量改进统计 ===")
    print(f"最大误差 - 有误差路径: {np.max(noisy_errors):.4f}")
    print(f"最大误差 - 校正后路径: {np.max(corrected_errors):.4f}")
    max_improvement = ((np.max(noisy_errors) - np.max(corrected_errors)) / np.max(noisy_errors) * 100)
    print(f"最大误差减少: {max_improvement:.2f}%")
    print(f"误差标准差 - 有误差路径: {np.std(noisy_errors):.4f}")
    print(f"误差标准差 - 校正后路径: {np.std(corrected_errors):.4f}")
    std_improvement = ((np.std(noisy_errors) - np.std(corrected_errors)) / np.std(noisy_errors) * 100)
    print(f"误差标准差减少: {std_improvement:.2f}%")
    
    plt.show()

if __name__ == "__main__":
    main()