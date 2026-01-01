import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_and_process_data(data_path='../data/printer_displacement_data.csv'):
    """
    加载并处理3D打印转角误差数据
    """
    print("加载数据...")
    
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        # 尝试其他可能的路径
        possible_paths = [
            '../data/printer_displacement_data.csv',
            'data/printer_displacement_data.csv',
            './data/printer_displacement_data.csv',
            'f:/AI/3D_printer_loss_perdict/data/printer_displacement_data.csv'
        ]
        
        found_path = None
        for path in possible_paths:
            if os.path.exists(path):
                found_path = path
                break
        
        if found_path is None:
            raise FileNotFoundError(f"数据文件不存在于任何已知路径: {possible_paths}")
        
        data_path = found_path
    
    # 读取CSV数据
    df = pd.read_csv(data_path)
    
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 分离特征和标签
    # 特征包括: velocity, radius, weight, stiffness, ideal_x, ideal_y
    feature_columns = ['velocity', 'radius', 'weight', 'stiffness', 'ideal_x', 'ideal_y']
    label_columns = ['displacement_x', 'displacement_y']
    
    X = df[feature_columns].values
    y = df[label_columns].values
    
    print(f"特征矩阵形状: {X.shape}")
    print(f"标签矩阵形状: {y.shape}")
    
    # 数据标准化
    print("标准化数据...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # 分割训练集和测试集
    print("分割训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    print(f"训练集形状 - X: {X_train.shape}, y: {y_train.shape}")
    print(f"测试集形状 - X: {X_test.shape}, y: {y_test.shape}")
    
    # 使用绝对路径保存标准化器
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(project_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    import joblib
    scaler_x_path = os.path.join(models_dir, 'scaler_X.pkl')
    scaler_y_path = os.path.join(models_dir, 'scaler_y.pkl')
    joblib.dump(scaler_X, scaler_x_path)
    joblib.dump(scaler_y, scaler_y_path)
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y

if __name__ == "__main__":
    # 确保模型目录存在
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(project_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 处理数据
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = load_and_process_data()
    
    print("数据处理完成!")
    print(f"训练样本数: {len(X_train)}")
    print(f"测试样本数: {len(X_test)}")