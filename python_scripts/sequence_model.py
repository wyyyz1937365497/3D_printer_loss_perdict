import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceDisplacementPredictor(nn.Module):
    """
    序列位移预测模型，输入一段路径序列，输出对应的位移向量序列
    """
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, output_size=2, sequence_length=10):
        """
        初始化模型
        :param input_size: 输入特征数 (velocity, radius, weight, stiffness)
        :param hidden_size: LSTM隐藏层大小
        :param num_layers: LSTM层数
        :param output_size: 输出维度 (x, y 位移)
        :param sequence_length: 序列长度
        """
        super(SequenceDisplacementPredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.sequence_length = sequence_length
        
        # LSTM层用于处理序列数据
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 全连接层用于从LSTM输出映射到位移向量
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Dropout层防止过拟合
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        前向传播
        :param x: 输入张量 (batch_size, sequence_length, input_size)
        :return: 输出张量 (batch_size, sequence_length, output_size)
        """
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 将LSTM输出通过全连接层
        batch_size, seq_len, _ = lstm_out.shape
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)
        
        # 全连接层
        out = F.relu(self.fc1(lstm_out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        # 重塑回序列格式
        out = out.view(batch_size, seq_len, self.output_size)
        
        return out