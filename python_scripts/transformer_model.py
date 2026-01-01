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


class TransformerDisplacementPredictor(nn.Module):
    """
    基于Transformer的位移预测模型
    """
    def __init__(self, input_size=4, d_model=128, nhead=8, num_layers=4, output_size=2, sequence_length=10, dropout=0.1):
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
        super(TransformerDisplacementPredictor, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size
        self.sequence_length = sequence_length
        
        # 线性投影层，将输入特征映射到d_model维度
        self.input_projection = nn.Linear(input_size, d_model)
        
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
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )
        
    def forward(self, x):
        """
        前向传播
        :param x: 输入张量 (batch_size, sequence_length, input_size)
        :return: 输出张量 (batch_size, sequence_length, output_size)
        """
        # 输入投影
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        output = self.transformer_encoder(x)
        
        # 输出投影
        output = self.output_projection(output)
        
        return output