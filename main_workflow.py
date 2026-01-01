import os
import subprocess
import sys
import argparse

def run_matlab_simulation():
    """运行MATLAB仿真"""
    print("正在运行MATLAB仿真...")
    
    # 检查MATLAB是否可用
    try:
        result = subprocess.run(['matlab', '-batch', 'exit'], capture_output=True, timeout=30)
        matlab_available = result.returncode == 0
    except FileNotFoundError:
        matlab_available = False
    
    if not matlab_available:
        print("警告: MATLAB未找到，跳过仿真步骤")
        return False
    
    # 运行MATLAB生成数据脚本
    matlab_script_path = os.path.join(os.path.dirname(__file__), 'matlab_simulation', 'generate_data.m')
    cmd = f'matlab -batch "cd {os.path.dirname(matlab_script_path)}; generate_data; exit"'
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"MATLAB仿真失败: {result.stderr}")
        return False
    
    print("MATLAB仿真成功完成")
    return True

def run_sequence_matlab_simulation():
    """运行序列数据的MATLAB仿真"""
    print("正在运行MATLAB序列数据仿真...")
    
    # 检查MATLAB是否可用
    try:
        result = subprocess.run(['matlab', '-batch', 'exit'], capture_output=True, timeout=30)
        matlab_available = result.returncode == 0
    except FileNotFoundError:
        matlab_available = False
    
    if not matlab_available:
        print("警告: MATLAB未找到，跳过序列仿真步骤")
        return False
    
    # 运行MATLAB生成序列数据脚本
    matlab_script_path = os.path.join(os.path.dirname(__file__), 'matlab_simulation', 'generate_sequence_data.m')
    cmd = f'matlab -batch "cd {os.path.dirname(matlab_script_path)}; generate_sequence_data; exit"'
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"MATLAB序列仿真失败: {result.stderr}")
        return False
    
    print("MATLAB序列仿真成功完成")
    return True

def run_python_processing():
    """运行Python数据处理"""
    print("正在运行Python数据处理...")
    
    # 导入并运行数据处理脚本
    try:
        from python_scripts import data_processing
        data_processing.process_data()
        print("Python数据处理成功完成")
        return True
    except ImportError as e:
        print(f"导入数据处理模块失败: {e}")
        return False
    except Exception as e:
        print(f"Python数据处理失败: {e}")
        return False

def run_python_training():
    """运行Python模型训练"""
    print("正在运行Python模型训练...")
    
    try:
        from python_scripts import train_model
        train_model.train_model()
        print("Python模型训练成功完成")
        return True
    except ImportError as e:
        print(f"导入模型训练模块失败: {e}")
        return False
    except Exception as e:
        print(f"Python模型训练失败: {e}")
        return False

def run_sequence_training():
    """运行序列模型训练"""
    print("正在运行序列模型训练...")
    
    try:
        from python_scripts import train_sequence_model
        train_sequence_model.train_sequence_model()
        print("序列模型训练成功完成")
        return True
    except ImportError as e:
        print(f"导入序列模型训练模块失败: {e}")
        return False
    except Exception as e:
        print(f"序列模型训练失败: {e}")
        return False

def run_python_apply():
    """运行Python模型应用"""
    print("正在应用训练好的模型...")
    
    try:
        from python_scripts import apply_model
        apply_model.apply_model()
        print("模型应用成功完成")
        return True
    except ImportError as e:
        print(f"导入模型应用模块失败: {e}")
        return False
    except Exception as e:
        print(f"Python模型应用失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='3D打印机转角误差补偿系统工作流程')
    parser.add_argument('--skip-matlab', action='store_true', help='跳过MATLAB仿真步骤')
    parser.add_argument('--skip-sequence', action='store_true', help='跳过序列模型步骤')
    
    args = parser.parse_args()
    
    print("开始执行3D打印机转角误差补偿系统工作流程...")
    
    steps = []
    steps.append(('Python数据处理', run_python_processing))
    steps.append(('Python模型训练', run_python_training))
    
    if not args.skip_matlab:
        steps.append(('MATLAB仿真', run_matlab_simulation))
    
    steps.append(('Python模型应用', run_python_apply))
    
    if not args.skip_matlab and not args.skip_sequence:
        steps.append(('MATLAB序列仿真', run_sequence_matlab_simulation))
        steps.append(('序列模型训练', run_sequence_training))
    
    successful_steps = 0
    total_steps = len(steps)
    
    for step_name, step_func in steps:
        print(f"正在执行: {step_name}")
        if step_func():
            successful_steps += 1
        else:
            print(f"步骤 '{step_name}' 失败")
    
    print(f"\n所有步骤执行完成！成功 {successful_steps}/{total_steps}")
    
    if successful_steps == total_steps:
        print("\n生成的文件:")
        print("- data/printer_displacement_data.csv: 仿真生成的数据")
        print("- models/displacement_predictor.pth: 训练好的模型")
        print("- models/sequence_displacement_predictor.pth: 序列模型")
        print("- models/scaler_X.pkl, models/scaler_y.pkl: 标准化器")
        print("- models/sequence_scaler_X.pkl, models/sequence_scaler_y.pkl: 序列标准化器")
        print("- models/training_results.png: 训练结果图")
        print("- models/sequence_training_results.png: 序列训练结果图")
    else:
        print(f"\n有 {total_steps - successful_steps} 个步骤失败，请检查错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main()