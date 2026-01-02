def generate_model_documentation():
    """
    生成交互式HTML文档
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>混合神经网络模型文档</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
            h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
            h2 { color: #555; margin-top: 30px; }
            .box { background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0; }
            .input-box { background: #E8F5E9; border-left: 4px solid #4CAF50; }
            .transform-box { background: #E3F2FD; border-left: 4px solid #2196F3; }
            .lstm-box { background: #FFF3E0; border-left: 4px solid #FF9800; }
            .output-box { background: #F3E5F5; border-left: 4px solid #9C27B0; }
            .arrow { text-align: center; font-size: 24px; color: #666; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background: #4CAF50; color: white; }
            tr:nth-child(even) { background: #f9f9f9; }
        </style>
    </head>
    <body>
        <h1>🧠 混合神经网络模型 (Transformer + LSTM)</h1>
        
        <h2>模型概述</h2>
        <div class="box">
            本模型结合了Transformer的全局注意力机制和LSTM的时序建模能力，
            用于预测打印过程中的位移偏差。模型综合考虑了速度、半径、重量、
            刚度等特征，以及速度方向和下一点位置信息。
        </div>
        
        <h2>架构流程</h2>
        <div class="box input-box">
            <strong>输入层</strong><br>
            基础特征: velocity, radius, weight, stiffness (4维)
        </div>
        <div class="arrow">↓</div>
        <div class="box input-box">
            <strong>特征扩展</strong><br>
            添加: 速度方向向量(2维) + 下一点位置(2维)<br>
            总计: 8维特征
        </div>
        <div class="arrow">↓</div>
        <div class="box input-box">
            <strong>线性投影</strong><br>
            Linear(8 → 128) × √128
        </div>
        <div class="arrow">↓</div>
        <div class="box input-box">
            <strong>位置编码</strong><br>
            PositionalEncoding (序列长度: 10)
        </div>
        <div class="arrow">↓</div>
        <div class="box transform-box">
            <strong>Transformer编码器</strong><br>
            - 多头注意力机制 (8头)<br>
            - 前馈神经网络<br>
            - 层数: 2<br>
            - 维度: 128
        </div>
        <div class="arrow">↓</div>
        <div class="box lstm-box">
            <strong>双向LSTM</strong><br>
            - 输入维度: 128<br>
            - 隐藏层维度: 64<br>
            - 层数: 2<br>
            - 双向: 是
        </div>
        <div class="arrow">↓</div>
        <div class="box output-box">
            <strong>输出投影</strong><br>
            Linear(128 → 64) → ReLU → Dropout → Linear(64 → 2)
        </div>
        <div class="arrow">↓</div>
        <div class="box output-box">
            <strong>输出层</strong><br>
            displacement_x, displacement_y (2维)
        </div>
        
        <h2>参数配置</h2>
        <table>
            <tr>
                <th>参数</th>
                <th>值</th>
                <th>说明</th>
            </tr>
            <tr>
                <td>input_size</td>
                <td>4</td>
                <td>输入特征数量</td>
            </tr>
            <tr>
                <td>d_model</td>
                <td>128</td>
                <td>Transformer模型维度</td>
            </tr>
            <tr>
                <td>nhead</td>
                <td>8</td>
                <td>多头注意力头数</td>
            </tr>
            <tr>
                <td>num_layers</td>
                <td>2</td>
                <td>Transformer/LSTM层数</td>
            </tr>
            <tr>
                <td>output_size</td>
                <td>2</td>
                <td>输出维度 (x, y位移)</td>
            </tr>
            <tr>
                <td>sequence_length</td>
                <td>10</td>
                <td>输入序列长度</td>
            </tr>
            <tr>
                <td>dropout</td>
                <td>0.1</td>
                <td>Dropout概率</td>
            </tr>
        </table>
        
        <h2>输入输出说明</h2>
        <table>
            <tr>
                <th>类别</th>
                <th>维度</th>
                <th>说明</th>
            </tr>
            <tr>
                <td>输入特征 (x)</td>
                <td>(batch, 10, 4)</td>
                <td>velocity, radius, weight, stiffness</td>
            </tr>
            <tr>
                <td>速度方向 (velocities)</td>
                <td>(batch, 10, 2)</td>
                <td>x和y方向的速度分量</td>
            </tr>
            <tr>
                <td>下一点位置 (next_positions)</td>
                <td>(batch, 10, 2)</td>
                <td>下一个点的x, y坐标</td>
            </tr>
            <tr>
                <td>输出 (displacement)</td>
                <td>(batch, 10, 2)</td>
                <td>预测的displacement_x, displacement_y</td>
            </tr>
        </table>
        
    </body>
    </html>
    """
    
    with open('model_documentation.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("交互式文档已生成: model_documentation.html")

# 生成文档
generate_model_documentation()
