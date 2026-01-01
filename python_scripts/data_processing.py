import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def process_data():
    """
    处理数据的主要函数
    """
    print("加载数据...")
    
    # 确保数据目录存在
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    data_path = os.path.join(data_dir, 'printer_displacement_data.csv')
    
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        return False
    
    df = pd.read_csv(data_path)
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 提取特征和标签
    feature_cols = ['velocity', 'radius', 'weight', 'stiffness', 'ideal_x', 'ideal_y']
    label_cols = ['displacement_x', 'displacement_y']
    
    X = df[feature_cols].values
    y = df[label_cols].values
    
    print(f"特征矩阵形状: {X.shape}")
    print(f"标签矩阵形状: {y.shape}")
    
    # 标准化数据
    print("标准化数据...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # 分割训练集和测试集
    print("分割训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    print(f"训练集形状 - X: {X_train.shape}, y: {y_train.shape}")
    print(f"测试集形状 - X: {X_test.shape}, y: {y_test.shape}")
    
    print("数据处理完成!")
    
    # 返回处理后的数据
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y

if __name__ == "__main__":
    # 确保模型目录存在
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(project_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 处理数据
    result = process_data()
    
    if result and result is not False:
        X_train, X_test, y_train, y_test, scaler_X, scaler_y = result
        
        # 保存标准化器
        scaler_x_path = os.path.join(models_dir, 'scaler_X.pkl')
        scaler_y_path = os.path.join(models_dir, 'scaler_y.pkl')
        joblib.dump(scaler_X, scaler_x_path)
        joblib.dump(scaler_y, scaler_y_path)
        
        print(f"训练样本数: {len(X_train)}")
        print(f"测试样本数: {len(X_test)}")
        print(f"标准化器已保存到: {models_dir}")