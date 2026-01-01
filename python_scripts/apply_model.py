import torch
import joblib
import numpy as np
import os

# 定义与训练时相同的模型结构
class DisplacementPredictor(torch.nn.Module):
    """
    位移预测模型（与train_model.py中的定义相同）
    """
    def __init__(self, input_size=6, hidden_sizes=[128, 256, 128], output_size=2):
        super(DisplacementPredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(prev_size, hidden_size))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(0.3))
            prev_size = hidden_size
        
        layers.append(torch.nn.Linear(prev_size, output_size))
        
        self.network = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def apply_model():
    """
    应用训练好的模型进行预测
    """
    print("测试模型...")
    
    # 加载模型和标准化器
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(project_dir, 'models')
    
    model_path = os.path.join(models_dir, 'displacement_predictor.pth')
    scaler_x_path = os.path.join(models_dir, 'scaler_X.pkl')
    scaler_y_path = os.path.join(models_dir, 'scaler_y.pkl')
    
    # 初始化并加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DisplacementPredictor()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 加载标准化器
    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    
    # 测试几个案例
    test_cases = [
        [100, 10, 10, 1.0, 5, 5],      # 速度100, 半径10, 重量10, 刚度1.0, 理想位置(5,5)
        [50, 20, 10, 1.2, 10, 10],     # 速度50, 半径20, 重量10, 刚度1.2, 理想位置(10,10)
        [150, 5, 10, 0.8, 15, 15],     # 速度150, 半径5, 重量10, 刚度0.8, 理想位置(15,15)
        [120, 1, 10, 1.0, 20, 20],     # 速度120, 半径1, 重量10, 刚度1.0, 理想位置(20,20)
        [80, 2, 10, 0.9, 25, 25]       # 速度80, 半径2, 重量10, 刚度0.9, 理想位置(25,25)
    ]
    
    print("预测结果:")
    for i, case in enumerate(test_cases):
        v, r, w, k, x, y = case
        input_data = np.array([[v, r, w, k, x, y]])
        input_scaled = scaler_X.transform(input_data)
        input_tensor = torch.FloatTensor(input_scaled).to(device)
        
        with torch.no_grad():
            prediction_scaled = model(input_tensor)
            prediction_scaled = prediction_scaled.cpu().numpy()
        
        prediction = scaler_y.inverse_transform(prediction_scaled)
        
        print(f"测试案例 {i+1}:")
        print(f"  输入参数: v={v}, r={r}, w={w}, k={k}")
        print(f"  理想位置: ({x}, {y})")
        print(f"  预测偏差: ({prediction[0][0]:.4f}, {prediction[0][1]:.4f})")
        print(f"  校正后位置: ({x - prediction[0][0]:.4f}, {y - prediction[0][1]:.4f})")
        print()
    
    # 模拟路径校正
    print("模型应用示例:")
    print("理想路径 -> 校正后路径，考虑转角误差补偿")
    
    # 创建一个简单的路径
    path = [(0, 0), (5, 5), (10, 10), (15, 5), (20, 0)]
    corrected_path = []
    
    print("路径校正结果:")
    for i, pt in enumerate(path):
        x, y = pt
        # 使用固定参数进行预测
        input_data = np.array([[100, 5, 10, 1.0, x, y]])  # 速度100, 半径5, 重量10, 刚度1.0
        input_scaled = scaler_X.transform(input_data)
        input_tensor = torch.FloatTensor(input_scaled).to(device)
        
        with torch.no_grad():
            prediction_scaled = model(input_tensor)
            prediction_scaled = prediction_scaled.cpu().numpy()
        
        prediction = scaler_y.inverse_transform(prediction_scaled)
        
        corrected_x = x - prediction[0][0]
        corrected_y = y - prediction[0][1]
        
        corrected_path.append((corrected_x, corrected_y))
        
        print(f"  点 {i}: ({x}, {y}) -> ({corrected_x:.4f}, {corrected_y:.4f})")
    
    print("\n应用完成!")

if __name__ == "__main__":
    apply_model()