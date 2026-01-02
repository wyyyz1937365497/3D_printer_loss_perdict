import torch
import torch.nn as nn
import math
import os

# 导入混合模型定义
from hybrid_model import HybridDisplacementPredictor

def create_model_and_run_inference():
    """
    创建混合模型并在GPU上进行推理演示，解决设备不匹配问题
    """
    print("检查CUDA是否可用:", torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化混合模型
    model = HybridDisplacementPredictor(
        input_size=4,  # velocity, radius, weight, stiffness
        d_model=128,
        nhead=8,
        num_layers=2,
        output_size=2,  # displacement_x, displacement_y
        sequence_length=10,
        dropout=0.1
    ).to(device)  # 将模型移动到指定设备
    
    # 创建测试输入数据 - 必须在相同设备上
    batch_size = 1
    sequence_length = 10
    
    # 基础输入特征 (batch_size, sequence_length, input_size)
    x = torch.randn(batch_size, sequence_length, 4).to(device)  # velocity, radius, weight, stiffness
    
    # 速度方向特征 (batch_size, sequence_length, 2)
    velocities = torch.randn(batch_size, sequence_length, 2).to(device)  # velocity_x, velocity_y
    
    # 下一点位置特征 (batch_size, sequence_length, 2)
    next_positions = torch.randn(batch_size, sequence_length, 2).to(device)  # next_pos_x, next_pos_y
    
    print(f"输入张量 x 设备: {x.device}")
    print(f"输入张量 velocities 设备: {velocities.device}")
    print(f"输入张量 next_positions 设备: {next_positions.device}")
    print(f"模型设备: {next(model.parameters()).device}")
    
    # 执行前向传播
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 不计算梯度
        output = model(x, velocities, next_positions)
    
    print(f"输出形状: {output.shape}")
    print(f"输出设备: {output.device}")
    print("推理成功完成！")
    
    return model, x, velocities, next_positions, output

def demonstrate_device_fix():
    """
    演示如何修复设备不匹配问题
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*60)
    print("演示如何修复设备不匹配问题")
    print("="*60)
    
    # 初始化模型并移动到GPU
    model = HybridDisplacementPredictor(
        input_size=4,
        d_model=128,
        nhead=8,
        num_layers=2,
        output_size=2,
        sequence_length=10
    ).to(device)
    
    # 错误示例：输入张量在CPU上，模型在GPU上
    print("\n错误示例（会导致设备不匹配错误）:")
    print("- 模型设备:", next(model.parameters()).device)
    
    # 创建在CPU上的输入
    x_cpu = torch.randn(1, 10, 4)  # 在CPU上
    velocities_cpu = torch.randn(1, 10, 2)  # 在CPU上
    next_positions_cpu = torch.randn(1, 10, 2)  # 在CPU上
    print("- 输入张量设备:", x_cpu.device)
    
    print("\n正确做法:")
    print("1. 确保所有输入张量移到与模型相同的设备上")
    
    # 正确的做法：将输入也移到GPU上
    x_gpu = x_cpu.to(device)
    velocities_gpu = velocities_cpu.to(device)
    next_positions_gpu = next_positions_cpu.to(device)
    
    print(f"   - 模型设备: {next(model.parameters()).device}")
    print(f"   - 输入x设备: {x_gpu.device}")
    print(f"   - 速度输入设备: {velocities_gpu.device}")
    print(f"   - 位置输入设备: {next_positions_gpu.device}")
    
    print("\n2. 运行模型推理")
    model.eval()
    with torch.no_grad():
        output = model(x_gpu, velocities_gpu, next_positions_gpu)
        print(f"   - 输出形状: {output.shape}")
        print(f"   - 输出设备: {output.device}")
    
    print("\n3. 或者使用model.to(device)方法将所有输入转换到模型所在设备")
    # 例如: x = x_cpu.to(next(model.parameters()).device)

def load_trained_model_and_infer(model_path=None):
    """
    加载预训练模型并进行推理（如果存在）
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 检查是否有预训练模型
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(project_dir, 'models')
    
    if model_path is None:
        model_path = os.path.join(models_dir, 'hybrid_displacement_predictor.pth')
    
    if os.path.exists(model_path):
        print(f"\n加载预训练模型: {model_path}")
        
        # 重新初始化模型结构
        model = HybridDisplacementPredictor(
            input_size=4,
            d_model=128,
            nhead=8,
            num_layers=2,
            output_size=2,
            sequence_length=10
        )
        
        # 加载权重
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)  # 移动到目标设备
        
        # 创建测试输入
        x = torch.randn(1, 10, 4).to(device)
        velocities = torch.randn(1, 10, 2).to(device)
        next_positions = torch.randn(1, 10, 2).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(x, velocities, next_positions)
        
        print(f"使用预训练模型推理成功，输出形状: {output.shape}")
        return model
    else:
        print(f"\n未找到预训练模型: {model_path}")
        print("请先运行 train_hybrid_model.py 进行模型训练")
        return None

if __name__ == "__main__":
    print("修复混合模型GPU设备不匹配问题")
    print("="*50)
    
    # 演示模型创建和推理
    try:
        model, x, velocities, next_positions, output = create_model_and_run_inference()
    except Exception as e:
        print(f"推理过程中出现错误: {e}")
    
    # 演示修复方法
    demonstrate_device_fix()
    
    # 尝试加载预训练模型
    trained_model = load_trained_model_and_infer()