import torch
from torchviz import make_dot
from hybrid_model import HybridDisplacementPredictor

# 创建模型和虚拟输入
model = HybridDisplacementPredictor(
    input_size=4,
    d_model=128,
    nhead=8,
    num_layers=2,
    output_size=2,
    sequence_length=10
)

# 创建示例输入
batch_size = 2
x = torch.randn(batch_size, 10, 4)
velocities = torch.randn(batch_size, 10, 2)
next_positions = torch.randn(batch_size, 10, 2)

# 前向传播
output = model(x, velocities, next_positions)

# 生成计算图
graph = make_dot(output, params=dict(model.named_parameters()))
graph.render('hybrid_model_graph', format='png', cleanup=True)
print("计算图已保存为 hybrid_model_graph.png")
