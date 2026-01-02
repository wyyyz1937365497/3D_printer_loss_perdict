import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from hybrid_model import HybridDisplacementPredictor

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def draw_model_architecture():
    """
    绘制混合神经网络架构图
    """
    # 创建模型实例
    model = HybridDisplacementPredictor(
        input_size=4,
        d_model=128,
        nhead=8,
        num_layers=2,
        output_size=2,
        sequence_length=10,
        dropout=0.1
    )
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 颜色定义
    input_color = '#4CAF50'    # 绿色
    transform_color = '#2196F3'  # 蓝色
    lstm_color = '#FF9800'     # 橙色
    output_color = '#9C27B0'   # 紫色
    
    # 绘制标题
    ax.text(7, 9.5, '混合神经网络架构 (Transformer + LSTM)', 
            fontsize=20, ha='center', weight='bold')
    
    # 输入层
    input_box = FancyBboxPatch((0.5, 7), 2.5, 1.5, boxstyle="round,pad=0.1", 
                               edgecolor='black', facecolor=input_color, alpha=0.7)
    ax.add_patch(input_box)
    ax.text(1.75, 7.75, '输入层', fontsize=12, ha='center', va='center', weight='bold', color='white')
    ax.text(1.75, 7.25, 'velocity\nradius\nweight\nstiffness', fontsize=9, ha='center', va='center', color='white')
    
    # 扩展输入
    extended_box = FancyBboxPatch((3.5, 7), 2.5, 1.5, boxstyle="round,pad=0.1",
                                   edgecolor='black', facecolor='#66BB6A', alpha=0.7)
    ax.add_patch(extended_box)
    ax.text(4.75, 7.75, '扩展特征', fontsize=12, ha='center', va='center', weight='bold', color='white')
    ax.text(4.75, 7.25, '+ 速度方向\n+ 下一点位置', fontsize=9, ha='center', va='center', color='white')
    
    # 线性投影
    proj_box = FancyBboxPatch((6.5, 7), 2.5, 1.5, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='#81C784', alpha=0.7)
    ax.add_patch(proj_box)
    ax.text(7.75, 7.75, '线性投影', fontsize=12, ha='center', va='center', weight='bold', color='white')
    ax.text(7.75, 7.25, 'Linear\n4+2+2 → 128', fontsize=9, ha='center', va='center', color='white')
    
    # 位置编码
    pos_box = FancyBboxPatch((9.5, 7), 2.5, 1.5, boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor='#A5D6A7', alpha=0.7)
    ax.add_patch(pos_box)
    ax.text(10.75, 7.75, '位置编码', fontsize=12, ha='center', va='center', weight='bold', color='white')
    ax.text(10.75, 7.25, 'Positional\nEncoding', fontsize=9, ha='center', va='center', color='white')
    
    # Transformer编码器
    transformer_box = FancyBboxPatch((1, 4.5), 4, 2, boxstyle="round,pad=0.1",
                                     edgecolor='black', facecolor=transform_color, alpha=0.7)
    ax.add_patch(transformer_box)
    ax.text(3, 5.8, 'Transformer编码器', fontsize=14, ha='center', va='center', weight='bold', color='white')
    ax.text(3, 5.3, '多头注意力', fontsize=11, ha='center', va='center', color='white')
    ax.text(3, 4.8, '前馈神经网络', fontsize=11, ha='center', va='center', color='white')
    ax.text(3, 4.3, '层数: 2 | 头数: 8', fontsize=9, ha='center', va='center', color='white')
    
    # LSTM层
    lstm_box = FancyBboxPatch((6, 4.5), 3, 2, boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor=lstm_color, alpha=0.7)
    ax.add_patch(lstm_box)
    ax.text(7.5, 5.8, '双向LSTM', fontsize=14, ha='center', va='center', weight='bold', color='white')
    ax.text(7.5, 5.3, '输入: 128维', fontsize=11, ha='center', va='center', color='white')
    ax.text(7.5, 4.8, '隐藏层: 64维', fontsize=11, ha='center', va='center', color='white')
    ax.text(7.5, 4.3, '层数: 2 | 双向', fontsize=9, ha='center', va='center', color='white')
    
    # 输出投影
    output_proj_box = FancyBboxPatch((10, 4.5), 3, 2, boxstyle="round,pad=0.1",
                                     edgecolor='black', facecolor='#FFB74D', alpha=0.7)
    ax.add_patch(output_proj_box)
    ax.text(11.5, 5.8, '输出投影', fontsize=14, ha='center', va='center', weight='bold', color='white')
    ax.text(11.5, 5.3, '128 → 64', fontsize=11, ha='center', va='center', color='white')
    ax.text(11.5, 4.8, '64 → 2', fontsize=11, ha='center', va='center', color='white')
    ax.text(11.5, 4.3, 'ReLU + Dropout', fontsize=9, ha='center', va='center', color='white')
    
    # 输出层
    output_box = FancyBboxPatch((5, 1.5), 4, 1.5, boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor=output_color, alpha=0.7)
    ax.add_patch(output_box)
    ax.text(7, 2.25, '输出层', fontsize=12, ha='center', va='center', weight='bold', color='white')
    ax.text(7, 1.75, 'displacement_x\ndisplacement_y', fontsize=9, ha='center', va='center', color='white')
    
    # 绘制箭头
    def draw_arrow(x1, y1, x2, y2, color='black'):
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', 
                                mutation_scale=20, color=color, linewidth=2)
        ax.add_patch(arrow)
    
    # 连接箭头
    draw_arrow(3, 7, 3, 6.5)  # 输入到Transformer
    draw_arrow(5.5, 7.5, 5.5, 6.5)  # 扩展到Transformer
    draw_arrow(7.75, 7, 7.75, 6.5)  # 投影到Transformer/LSTM
    draw_arrow(11.5, 7, 11.5, 6.5)  # 位置编码到LSTM
    draw_arrow(5, 5.5, 6, 5.5)  # Transformer到LSTM
    draw_arrow(9, 5.5, 10, 5.5)  # LSTM到输出投影
    draw_arrow(11.5, 4.5, 11.5, 3)  # 输出投影到输出
    
    # 添加尺寸标注 - 现在model变量已定义
    ax.text(7, 3.5, f'总参数量: {sum(p.numel() for p in model.parameters()):,}', 
            fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 图例
    legend_elements = [
        mpatches.Patch(color=input_color, label='输入处理'),
        mpatches.Patch(color=transform_color, label='Transformer'),
        mpatches.Patch(color=lstm_color, label='LSTM'),
        mpatches.Patch(color=output_color, label='输出'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('hybrid_model_architecture.png', dpi=300, bbox_inches='tight')
    print("模型架构图已保存为 hybrid_model_architecture.png")
    plt.show()

# 运行绘图
draw_model_architecture()