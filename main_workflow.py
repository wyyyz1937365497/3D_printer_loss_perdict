import os
import subprocess
import sys
import argparse

def run_matlab_simulation():
    """运行MATLAB数据生成脚本"""
    print("正在运行MATLAB仿真以生成数据...")
    
    # 检查MATLAB是否可用
    try:
        result = subprocess.run(['matlab', '-batch', 'cd matlab_simulation; generate_data; exit'], 
                               capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("MATLAB仿真成功完成")
            print(result.stdout)
        else:
            print("MATLAB仿真出错:")
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("未找到MATLAB命令行工具，尝试使用matlab-cli（如果已安装）")
        try:
            result = subprocess.run(['matlab-cli', '-batch', 'cd matlab_simulation; generate_data; exit'], 
                                   capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("MATLAB仿真成功完成")
                print(result.stdout)
            else:
                print("MATLAB仿真出错:")
                print(result.stderr)
                return False
        except FileNotFoundError:
            print("未找到MATLAB命令行工具，请确保已安装MATLAB并将其添加到系统路径中")
            print("跳过MATLAB仿真步骤，假定数据文件已存在")
            # 检查数据文件是否存在
            data_file = "data/printer_displacement_data.csv"
            if not os.path.exists(data_file):
                print(f"错误: 数据文件 {data_file} 不存在")
                return False
    
    return True

def run_python_data_processing():
    """运行Python数据处理脚本"""
    print("正在运行Python数据处理...")
    
    # 确保所需的Python包已安装
    required_packages = ['pandas', 'numpy', 'scikit-learn', 'torch', 'matplotlib']
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"正在安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # 创建数据目录
    os.makedirs('data', exist_ok=True)
    
    # 运行数据处理脚本
    try:
        result = subprocess.run([sys.executable, "python_scripts/data_processing.py"], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("Python数据处理成功完成")
            print(result.stdout)
        else:
            print("Python数据处理出错:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"运行数据处理脚本时出错: {e}")
        return False
    
    return True

def run_python_training():
    """运行Python模型训练脚本"""
    print("正在运行Python模型训练...")
    
    try:
        result = subprocess.run([sys.executable, "python_scripts/train_model.py"], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("Python模型训练成功完成")
            print(result.stdout)
        else:
            print("Python模型训练出错:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"运行训练脚本时出错: {e}")
        return False
    
    return True

def run_model_application():
    """运行模型应用脚本"""
    print("正在应用训练好的模型...")
    
    try:
        result = subprocess.run([sys.executable, "python_scripts/apply_model.py"], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("模型应用成功完成")
            print(result.stdout)
        else:
            print("模型应用出错:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"运行应用脚本时出错: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='3D打印机转角误差补偿系统')
    parser.add_argument('--skip-matlab', action='store_true', 
                        help='跳过MATLAB仿真步骤')
    parser.add_argument('--skip-processing', action='store_true', 
                        help='跳过数据处理步骤')
    parser.add_argument('--skip-training', action='store_true', 
                        help='跳过模型训练步骤')
    parser.add_argument('--skip-application', action='store_true', 
                        help='跳过模型应用步骤')
    
    args = parser.parse_args()
    
    print("开始执行3D打印机转角误差补偿系统工作流程...")
    
    success = True
    
    if not args.skip_matlab:
        success &= run_matlab_simulation()
        if not success:
            print("MATLAB仿真失败，退出流程")
            return
    
    if not args.skip_processing:
        success &= run_python_data_processing()
        if not success:
            print("Python数据处理失败，退出流程")
            return
    
    if not args.skip_training:
        success &= run_python_training()
        if not success:
            print("Python模型训练失败，退出流程")
            return
    
    if not args.skip_application:
        success &= run_model_application()
        if not success:
            print("模型应用失败，退出流程")
            return
    
    if success:
        print("\n所有步骤成功完成！")
        print("生成的文件:")
        print("- data/printer_displacement_data.csv: 仿真生成的数据")
        print("- models/displacement_predictor.pth: 训练好的模型")
        print("- models/scaler_X.pkl, models/scaler_y.pkl: 标准化器")
        print("- models/training_results.png: 训练结果图")
    else:
        print("\n工作流程执行失败！")

if __name__ == "__main__":
    main()