import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    位置编码模块，用于给序列添加位置信息
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class HybridDisplacementPredictor(nn.Module):
    """
    结合RNN和Transformer的混合模型
    考虑当前点的速度方向和下一点位置对误差的影响
    """
    def __init__(self, input_size=4, d_model=128, nhead=8, num_layers=2, output_size=2, sequence_length=10, dropout=0.1):
        """
        初始化模型
        :param input_size: 输入特征数 (velocity, radius, weight, stiffness)
        :param d_model: Transformer模型维度
        :param nhead: 多头注意力头数
        :param num_layers: Transformer层数
        :param output_size: 输出维度 (x, y 位移)
        :param sequence_length: 序列长度
        :param dropout: Dropout概率
        """
        super(HybridDisplacementPredictor, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size
        self.sequence_length = sequence_length
        
        # 扩展输入尺寸，包含速度方向和下一点位置信息
        extended_input_size = input_size + 2 + 2  # 原始输入 + 速度方向向量 + 下一点位置
        
        # 线性投影层，将扩展输入特征映射到d_model维度
        self.input_projection = nn.Linear(extended_input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, sequence_length)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 双向LSTM层，增强时序建模能力
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层 - 修正维度：双向LSTM的输出是d_model，因为2*(d_model//2) = d_model（当d_model是偶数时）
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # 双向LSTM输出是d_model维度
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )
        
    def forward(self, x, velocities, next_positions):
        """
        前向传播
        :param x: 基础输入张量 (batch_size, sequence_length, input_size)
        :param velocities: 速度方向张量 (batch_size, sequence_length, 2) - x和y方向的速度
        :param next_positions: 下一点位置张量 (batch_size, sequence_length, 2) - x和y坐标
        :return: 输出张量 (batch_size, sequence_length, output_size)
        """
        # 合并输入特征
        combined_input = torch.cat([x, velocities, next_positions], dim=-1)
        
        # 输入投影
        projected = self.input_projection(combined_input) * math.sqrt(self.d_model)
        
        # 添加位置编码
        encoded = self.pos_encoder(projected)
        
        # Transformer编码
        transformer_output = self.transformer_encoder(encoded)
        
        # LSTM处理
        lstm_output, _ = self.lstm(transformer_output)
        
        # 输出投影
        output = self.output_projection(lstm_output)
        
        return output