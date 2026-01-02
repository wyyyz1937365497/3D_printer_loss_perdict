import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
import numpy as np
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def draw_inference_flow():
    """
    绘制混合模型推理流程图
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 定义节点位置
    input_pos = [(2, 8), (2, 6), (2, 4)]
    process_pos = [(6, 6)]
    output_pos = [(10, 6)]
    
    # 输入节点
    ax.text(2, 8.2, '基础特征', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(2, 8, 'velocity, radius,\nweight, stiffness', ha='center', va='center', fontsize=10)
    ax.add_patch(patches.Rectangle((1, 7.5), 2, 1, linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.7))
    
    ax.text(2, 6.2, '速度方向', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(2, 6, 'velocity_x, velocity_y', ha='center', va='center', fontsize=10)
    ax.add_patch(patches.Rectangle((1, 5.5), 2, 1, linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.7))
    
    ax.text(2, 4.2, '下一位置', ha='center', va='center', fontsize=12, weight='bold')
    ax.text(2, 4, 'next_pos_x, next_pos_y', ha='center', va='center', fontsize=10)
    ax.add_patch(patches.Rectangle((1, 3.5), 2, 1, linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.7))
    
    # 模型处理节点
    ax.text(6, 6, 'HybridDisplacementPredictor\n(Transformer + LSTM)', ha='center', va='center', 
            fontsize=12, weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    # 输出节点
    ax.text(10, 6, '位移预测\n(displacement_x, displacement_y)', ha='center', va='center', 
            fontsize=12, weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
    
    # 连接线
    # 从基础特征到模型
    ax.annotate('', xy=(5.5, 6), xytext=(3, 8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    ax.annotate('', xy=(5.5, 6), xytext=(3, 6),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    ax.annotate('', xy=(5.5, 6), xytext=(3, 4),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    
    # 从模型到输出
    ax.annotate('', xy=(9.5, 6), xytext=(6.5, 6),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    
    # 添加标签
    ax.text(1, 9.5, '推理流程', fontsize=16, weight='bold', ha='left')
    ax.text(4, 7.5, '特征融合', fontsize=10, ha='center', 
            bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.5))
    ax.text(8, 6.5, '预测', fontsize=10, ha='center', 
            bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.5))
    
    ax.set_xlim(0, 12)
    ax.set_ylim(2, 10)
    ax.axis('off')
    
    plt.title('混合模型推理流程图', fontsize=16, weight='bold', pad=20)
    return fig

def draw_training_flow():
    """
    绘制混合模型训练流程图
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 定义节点位置
    stages = [
        ("数据加载", (2, 8)),
        ("数据预处理", (2, 6)),
        ("模型初始化", (2, 4)),
        ("前向传播", (6, 6)),
        ("损失计算", (8, 6)),
        ("反向传播", (10, 6)),
        ("参数更新", (10, 4)),
        ("验证评估", (6, 4))
    ]
    
    # 绘制训练阶段
    for i, (label, pos) in enumerate(stages):
        x, y = pos
        if label in ["数据加载", "数据预处理", "模型初始化"]:
            ax.add_patch(patches.Ellipse((x, y), 2, 0.8, linewidth=2, edgecolor='purple', facecolor='plum', alpha=0.7))
        elif label in ["前向传播", "损失计算"]:
            ax.add_patch(patches.Rectangle((x-1, y-0.4), 2, 0.8, linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.7))
        elif label in ["反向传播", "参数更新"]:
            ax.add_patch(patches.Rectangle((x-1, y-0.4), 2, 0.8, linewidth=2, edgecolor='red', facecolor='lightcoral', alpha=0.7))
        elif label == "验证评估":
            ax.add_patch(patches.Ellipse((x, y), 2, 0.8, linewidth=2, edgecolor='orange', facecolor='moccasin', alpha=0.7))
        
        ax.text(x, y, label, ha='center', va='center', fontsize=10, weight='bold')
    
    # 连接线
    connections = [
        (0, 1), (1, 2),  # 数据阶段
        (2, 3), (3, 4), (4, 5), (5, 6),  # 训练循环
        (6, 3), (3, 7)  # 循环和验证
    ]
    
    for start, end in connections:
        start_pos = stages[start][1]
        end_pos = stages[end][1]
        ax.annotate('', xy=(end_pos[0]-0.8, end_pos[1]), xytext=(start_pos[0]+0.8, start_pos[1]),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', connectionstyle="arc3,rad=0.1"))
    
    # 添加epoch循环标签
    ax.text(8, 2.5, 'Epoch循环', fontsize=12, weight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # 绘制循环箭头
    arc = patches.FancyArrowPatch((10.5, 4), (10.5, 6.5),
                                  connectionstyle="arc3,rad=0.5",
                                  arrowstyle='->', mutation_scale=20, color='blue', lw=2)
    ax.add_patch(arc)
    
    ax.text(11.5, 5.5, '迭代', rotation=-90, fontsize=10, weight='bold')
    
    ax.set_xlim(0, 12)
    ax.set_ylim(1, 9)
    ax.axis('off')
    
    plt.title('混合模型训练流程图', fontsize=16, weight='bold', pad=20)
    return fig

def main():
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # 绘制推理流程图
    fig1 = draw_inference_flow()
    inference_path = os.path.join(project_dir, 'hybrid_model_inference_flow.png')
    fig1.savefig(inference_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"推理流程图已保存至: {inference_path}")
    
    # 绘制训练流程图
    fig2 = draw_training_flow()
    training_path = os.path.join(project_dir, 'hybrid_model_training_flow.png')
    fig2.savefig(training_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"训练流程图已保存至: {training_path}")

if __name__ == "__main__":
    main()