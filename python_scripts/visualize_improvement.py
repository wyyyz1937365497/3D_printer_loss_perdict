import numpy as np
import matplotlib.pyplot as plt
import torch
import joblib
import os
from train_model import DisplacementPredictor

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

def generate_detailed_ideal_path():
    """
    生成详细的理想打印路径，包含更密集的路径点，特别是转角处
    """
    path_points = []
    
    # 生成一个更复杂的路径，包含密集的路径点
    # 路径：从原点出发，画一个矩形，然后画一个星形
    
    # 矩形路径（更密集的点）
    rect_side_length = 20
    # 底边
    for i in np.linspace(0, rect_side_length, 20):
        path_points.append((i, 0))
    # 右边
    for i in np.linspace(0, rect_side_length, 20):
        path_points.append((rect_side_length, i))
    # 顶边
    for i in np.linspace(rect_side_length, 0, 20):
        path_points.append((i, rect_side_length))
    # 左边
    for i in np.linspace(rect_side_length, 0, 20):
        path_points.append((0, i))
    
    # 生成星形路径
    star_center = (30, 10)
    outer_radius = 8
    inner_radius = 4
    num_points = 5
    
    # 计算星形的点
    for i in range(num_points):
        # 外部点
        angle = i * 2 * np.pi / num_points - np.pi/2  # 从顶部开始
        x = star_center[0] + outer_radius * np.cos(angle)
        y = star_center[1] + outer_radius * np.sin(angle)
        path_points.append((x, y))
        
        # 内部点
        inner_angle = (i + 0.5) * 2 * np.pi / num_points - np.pi/2
        x = star_center[0] + inner_radius * np.cos(inner_angle)
        y = star_center[1] + inner_radius * np.sin(inner_angle)
        path_points.append((x, y))
    
    # 将星形闭合
    path_points.append(path_points[80])  # 闭合星形，星形开始于索引80
    
    return path_points

def calculate_curvature_at_point(path, idx):
    """
    计算路径上某点的曲率，用于判断是否为转角
    """
    if idx == 0 or idx == len(path) - 1:
        # 端点使用相邻点计算
        if idx == 0:
            p1, p2 = path[0], path[1]
            # 使用前两个点的方向作为参考
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # 曲率近似为0
            return 0.0, np.arctan2(dy, dx)
        else:  # idx == len(path) - 1
            p1, p2 = path[-2], path[-1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            return 0.0, np.arctan2(dy, dx)
    
    # 使用三点法计算曲率
    prev_pt = path[idx - 1]
    curr_pt = path[idx]
    next_pt = path[idx + 1]
    
    # 计算两条线段的向量
    vec1 = (curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1])
    vec2 = (next_pt[0] - curr_pt[0], next_pt[1] - curr_pt[1])
    
    # 计算线段长度
    len1 = np.sqrt(vec1[0]**2 + vec1[1]**2)
    len2 = np.sqrt(vec2[0]**2 + vec2[1]**2)
    
    # 避免除以零
    if len1 < 1e-6 or len2 < 1e-6:
        return 0.0, 0.0
    
    # 计算单位向量
    unit_vec1 = (vec1[0] / len1, vec1[1] / len1)
    unit_vec2 = (vec2[0] / len2, vec2[1] / len2)
    
    # 计算角度变化
    angle_change = np.arccos(np.clip(unit_vec1[0] * unit_vec2[0] + unit_vec1[1] * unit_vec2[1], -1.0, 1.0))
    
    # 计算曲率（角度变化除以路径长度的一半）
    avg_len = (len1 + len2) / 2
    curvature = angle_change / avg_len if avg_len > 1e-6 else 0.0
    
    # 计算切线方向（平均方向）
    tangent_angle = np.arctan2((unit_vec1[1] + unit_vec2[1])/2, (unit_vec1[0] + unit_vec2[0])/2)
    
    return curvature, tangent_angle

def simulate_physical_errors_with_matlab_logic(ideal_path, velocities):
    """
    使用与MATLAB相同的逻辑模拟物理因素导致的打印误差
    """
    noisy_path = []
    
    for i, (x, y) in enumerate(ideal_path):
        if i < len(velocities):
            v = velocities[i]
        else:
            v = velocities[-1]  # 如果速度不够，使用最后一个速度
            
        # 计算当前点的曲率
        curvature, tangent_angle = calculate_curvature_at_point(ideal_path, i)
        
        # 将曲率转换为半径（避免除以零）
        if curvature > 1e-6:
            radius = 1.0 / curvature
        else:
            radius = 1000.0  # 直线段
        
        # 确保半径在合理范围内（与MATLAB相同）
        if radius < 0.5:
            radius = 0.5
        elif radius > 50:
            radius = 50.0
        
        # 固定参数（与MATLAB相同）
        nozzle_weight = 10  # 喷头重量
        # 使用固定的刚度值或根据路径点位置生成确定性刚度值
        k = 0.5 + (x + y) % 1.5  # 基于位置生成确定性刚度值
        
        # 模拟MATLAB中的物理逻辑
        # 对于尖锐转角和圆角使用不同的角度处理
        if radius <= 5:  # 尖锐转角
            angle = np.pi/2  # 90度转角
        else:  # 圆角
            # 使用切线方向作为参考角度
            angle = tangent_angle if abs(tangent_angle) > 1e-6 else np.pi/4
        
        # 计算实际偏差（基于物理模型的简化模拟，与MATLAB相同）
        # 偏差与速度平方、喷头重量成正比，与支撑梁刚度成反比
        # 尖锐转角通常会有更大的偏差
        base_deviation = (v**2 * nozzle_weight) / (k * 1000)
        
        # 尖锐转角的偏差更大
        if radius <= 5:
            deviation_factor = 1.5  # 尖锐转角偏差放大因子
        else:
            deviation_factor = 1.0  # 圆角偏差正常
        
        deviation_x = base_deviation * deviation_factor * np.cos(angle + np.pi/4)
        deviation_y = base_deviation * deviation_factor * np.sin(angle + np.pi/4)
        
        # 添加随机噪声模拟其他因素（与MATLAB相同）
        noise_factor = 0.1
        deviation_x = deviation_x + noise_factor * np.random.randn()
        deviation_y = deviation_y + noise_factor * np.random.randn()
        
        # 应用偏差到理想位置
        noisy_x = x + deviation_x
        noisy_y = y + deviation_y
        
        noisy_path.append((noisy_x, noisy_y))
    
    return noisy_path

def apply_model_correction(ideal_path, noisy_path, velocities, model, scaler_X, scaler_y, corner_threshold=0.1):
    """
    应用训练好的模型对有误差的路径进行修正
    只对曲率超过阈值的转角点进行修正
    """
    corrected_path = []
    
    for i, (ideal_pt, noisy_pt) in enumerate(zip(ideal_path, noisy_path)):
        if i < len(velocities):
            v = velocities[i]
        else:
            v = velocities[-1]  # 如果速度不够，使用最后一个速度
            
        # 计算当前点的曲率
        curvature, tangent_angle = calculate_curvature_at_point(ideal_path, i)
        
        # 将曲率转换为半径
        if curvature > 1e-6:
            radius = 1.0 / curvature
        else:
            radius = 1000.0  # 直线段
        
        # 确保半径在合理范围内
        if radius < 1:
            radius = 1
        elif radius > 1000:
            radius = 1000
        
        # 固定参数
        weight = 10  # 喷头重量
        # 使用与模拟时相同的确定性刚度值生成方法
        x_ideal, y_ideal = ideal_pt
        stiffness = 0.5 + (x_ideal + y_ideal) % 1.5  # 基于位置生成确定性刚度值
        
        # 准备输入数据 - 使用理想位置（因为模型训练时是用理想位置作为输入的）
        input_data = np.array([[v, radius, weight, stiffness, x_ideal, y_ideal]])
        
        # 标准化输入
        input_scaled = scaler_X.transform(input_data)
        
        # 转换为PyTorch张量
        input_tensor = torch.FloatTensor(input_scaled)
        
        # 预测偏差（理想位置到实际位置的偏差）
        with torch.no_grad():
            prediction_scaled = model(input_tensor)
            prediction_scaled = prediction_scaled.numpy()
        
        # 反标准化预测结果
        prediction = scaler_y.inverse_transform(prediction_scaled)
        
        # 模型预测的是从理想位置到实际位置的偏差（ideal -> actual）
        # 因此，要从实际位置（有误差的位置）减去这个偏差，来接近理想位置
        # corrected = actual - (actual - ideal) = ideal
        x_noisy, y_noisy = noisy_pt
        
        # 只对转角点进行修正（曲率大于阈值）
        if curvature > corner_threshold:
            # 根据经验教训中的"模型预测偏差方向的应用与验证原则"，确认校正方向
            corrected_x = x_noisy - prediction[0][0]  # 从有误差的位置减去预测的偏差
            corrected_y = y_noisy - prediction[0][1]
        else:
            # 非转角点保持不变
            corrected_x = x_noisy
            corrected_y = y_noisy
        
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
    
    # 生成详细的理想路径
    ideal_path = generate_detailed_ideal_path()
    print(f"生成了 {len(ideal_path)} 个路径点")
    
    # 定义速度（模拟不同段的不同速度）
    velocities = [50 if i < 80 else 80 if i < 160 else 60 if i < 200 else 100 for i in range(len(ideal_path))]
    
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    
    # 模拟物理误差（使用与MATLAB相同的逻辑）
    noisy_path = simulate_physical_errors_with_matlab_logic(ideal_path, velocities)
    
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
    
    # 应用模型校正（对有误差的路径进行校正，仅对转角点）
    corrected_path = apply_model_correction(ideal_path, noisy_path, velocities, model, scaler_X, scaler_y)
    
    # 计算误差
    noisy_errors = calculate_errors(ideal_path, noisy_path)
    corrected_errors = calculate_errors(ideal_path, corrected_path)
    
    # 打印统计信息
    print(f"平均误差 - 理想路径: 0.0000")
    print(f"平均误差 - 有噪声路径: {np.mean(noisy_errors):.4f}")
    print(f"平均误差 - 校正后路径: {np.mean(corrected_errors):.4f}")
    improvement = ((np.mean(noisy_errors) - np.mean(corrected_errors)) / np.mean(noisy_errors) * 100)
    print(f"误差减少: {improvement:.2f}%")
    
    # 计算曲率用于可视化
    curvatures = []
    for i in range(len(ideal_path)):
        curvature, _ = calculate_curvature_at_point(ideal_path, i)
        curvatures.append(curvature)
    curvatures = np.array(curvatures)
    
    # 创建可视化图表
    plt.figure(figsize=(20, 5))
    
    # 子图1: 路径对比
    plt.subplot(1, 4, 1)
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
    plt.subplot(1, 4, 2)
    plt.plot(noisy_errors, 'r-', linewidth=2, label='有误差路径误差', alpha=0.8)
    plt.plot(corrected_errors, 'b-', linewidth=2, label='校正后路径误差', alpha=0.8)
    plt.title('路径点误差对比')
    plt.xlabel('路径点索引')
    plt.ylabel('误差 (mm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3: 误差分布直方图
    plt.subplot(1, 4, 3)
    plt.hist(noisy_errors, bins=30, alpha=0.6, label='有误差路径', color='red', density=True)
    plt.hist(corrected_errors, bins=30, alpha=0.6, label='校正后路径', color='blue', density=True)
    plt.title('误差分布对比')
    plt.xlabel('误差 (mm)')
    plt.ylabel('概率密度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图4: 曲率分布
    plt.subplot(1, 4, 4)
    plt.plot(curvatures, 'g-', linewidth=1, label='路径曲率', alpha=0.8)
    plt.title('路径曲率分布')
    plt.xlabel('路径点索引')
    plt.ylabel('曲率')
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
    
    # 计算转角点的改进
    corner_indices = np.where(curvatures > 0.1)[0]
    if len(corner_indices) > 0:
        corner_noisy_errors = noisy_errors[corner_indices]
        corner_corrected_errors = corrected_errors[corner_indices]
        corner_improvement = ((np.mean(corner_noisy_errors) - np.mean(corner_corrected_errors)) / np.mean(corner_noisy_errors) * 100)
        print(f"转角点平均误差改善: {corner_improvement:.2f}%")
    
    plt.show()

if __name__ == "__main__":
    main()