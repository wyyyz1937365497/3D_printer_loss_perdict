import torch
from torch.utils.tensorboard import SummaryWriter
from hybrid_model import HybridDisplacementPredictor
import os

def visualize_model_with_tensorboard():
    """
    使用TensorBoard可视化模型结构
    """
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建混合模型实例
    model = HybridDisplacementPredictor(
        input_size=4,  # velocity, radius, weight, stiffness
        d_model=128,
        nhead=8,
        num_layers=2,
        output_size=2,  # displacement_x, displacement_y
        sequence_length=10,
        dropout=0.1
    )
    
    # 将模型移动到指定设备
    model = model.to(device)
    
    # 创建一个SummaryWriter实例
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    writer = SummaryWriter(log_dir)
    
    # 创建示例输入 - 确保所有输入都在正确的设备上
    batch_size = 1
    sequence_length = 10
    
    # 创建在CPU上的示例输入
    x = torch.randn(batch_size, sequence_length, 4)
    velocities = torch.randn(batch_size, sequence_length, 2)
    next_positions = torch.randn(batch_size, sequence_length, 2)
    
    # 如果使用GPU，将输入也移动到GPU
    if device.type == 'cuda':
        x = x.to(device)
        velocities = velocities.to(device)
        next_positions = next_positions.to(device)
    
    # 准备模型输入元组
    dummy_input = (x, velocities, next_positions)
    
    print("输入张量形状:")
    print(f"  x: {x.shape}")
    print(f"  velocities: {velocities.shape}")
    print(f"  next_positions: {next_positions.shape}")
    
    try:
        # 尝试添加模型图到TensorBoard
        # 由于模型有多个输入，我们需要修改模型的forward方法以接受元组输入
        # 或者使用torch.jit.trace_module
        print("正在生成模型图...")
        
        # 定义一个包装类，将多个输入打包成一个输入
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x_combined):
                x, velocities, next_positions = x_combined
                return self.model(x, velocities, next_positions)
        
        wrapper_model = ModelWrapper(model)
        
        # 将输入包装成元组
        def forward_with_tuple_input(model_input_tuple):
            x, velocities, next_positions = model_input_tuple
            return model(x, velocities, next_positions)
        
        # 尝试添加图 - 使用eval模式避免dropout等随机性
        model.eval()
        
        # 使用torch.no_grad()避免梯度计算影响图结构
        with torch.no_grad():
            # 首先运行一次模型确保状态一致
            _ = model(x, velocities, next_positions)
            
            # 再次添加到TensorBoard
            writer.add_graph(model, dummy_input)
        
        print("模型图已成功添加到TensorBoard!")
        
    except torch.jit._trace.TracingCheckError as e:
        print(f"追踪模型时出错: {e}")
        print("这通常是由于模型中的动态行为导致的，正在尝试替代方法...")
        
        # 尝试使用eval模式和no_grad上下文
        model.eval()
        with torch.no_grad():
            try:
                # 简单地运行模型一次，但不追踪图
                output = model(x, velocities, next_positions)
                print(f"模型前向传播成功，输出形状: {output.shape}")
                
                # 手动创建一个简单的可视化
                print("由于追踪失败，将使用替代方法可视化模型结构...")
                print("\n模型结构:")
                print(model)
                
                print("\n模型参数统计:")
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"总参数数: {total_params:,}")
                print(f"可训练参数数: {trainable_params:,}")
                
            except Exception as e2:
                print(f"替代方法也失败了: {e2}")
    
    except Exception as e:
        print(f"添加模型图到TensorBoard时出错: {e}")
    
    finally:
        # 关闭writer
        writer.close()
        print(f"\nTensorBoard日志已保存到: {log_dir}")
        print("要查看可视化结果，请在终端运行: tensorboard --logdir=" + log_dir)

def print_model_info():
    """
    打印模型的详细信息
    """
    model = HybridDisplacementPredictor(
        input_size=4,
        d_model=128,
        nhead=8,
        num_layers=2,
        output_size=2,
        sequence_length=10,
        dropout=0.1
    )
    
    print("\n模型架构详情:")
    print(model)
    
    print("\n各层参数数量:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape} - {param.numel():,} 参数")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")

if __name__ == "__main__":
    print("开始TensorBoard模型可视化...")
    visualize_model_with_tensorboard()
    print("\n" + "="*50)
    print_model_info()