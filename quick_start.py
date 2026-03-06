#!/usr/bin/env python3
# -*- coding: utf- -*-
"""
茶色素智能检测系统 - 快速启动脚本
Quick Start Script for Tea Pigment Detection System

这个脚本用于快速启动系统，自动处理环境检查和依赖安装
"""

import sys
import os
import subprocess
from pathlib import Path


def print_header(title):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def check_python_version():
    """检查Python版本"""
    print_header("检查Python版本")
    
    version = sys.version_info
    print(f"当前Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✓ Python版本符合要求 (3.8+)")
        return True
    else:
        print("✗ Python版本不符合要求，需要 Python 3.8 或更高版本")
        print("请安装 Python 3.8+ 后重试")
        return False


def check_virtual_env():
    """检查是否在虚拟环境中"""
    print_header("检查虚拟环境")
    
    in_venv = (hasattr(sys, 'real_prefix') or 
               (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
    
    if in_venv:
        print("✓ 当前在虚拟环境中")
        venv_path = sys.prefix
        print(f"虚拟环境路径: {venv_path}")
        return True
    else:
        print("⚠ 不在虚拟环境中")
        print("建议创建虚拟环境后再运行（可选）")
        return True  # 不强制要求


def install_dependencies():
    """安装依赖包"""
    print_header("安装依赖包")
    
    # 检查 requirements.txt 是否存在
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        print(f"✗ 找不到 {requirements_file} 文件")
        return False
    
    print("正在安装依赖包，这可能需要几分钟...")
    
    try:
        # 升级 pip
        print("\n[1/2] 升级 pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                             stdout=subprocess.DEVNULL)
        print("✓ pip 升级完成")
        
        # 安装依赖
        print("\n[2/2] 安装依赖包...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("✓ 依赖包安装完成")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ 依赖安装失败: {e}")
        print("\n您可以手动执行以下命令安装依赖:")
        print(f"  pip install -r {requirements_file}")
        return False


def check_project_structure():
    """检查项目结构"""
    print_header("检查项目结构")
    
    required_files = [
        "main_enhanced.py",
        "config.py",
        "requirements.txt",
        "core/__init__.py",
        "core/image_processor.py",
        "core/pigment_predictor_fixed.py",
        "INSTALL.md"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (缺失)")
            all_exist = False
    
    if all_exist:
        print("\n✓ 所有必需文件都存在")
        return True
    else:
        print("\n✗ 部分必需文件缺失，请检查项目结构")
        return False


def check_models():
    """检查模型文件"""
    print_header("检查模型文件")
    
    model_dir = "models"
    model_files = [
        "tf_model.pkl",
        "tr_model.pkl",
        "tb_model.pkl",
        "scaler.pkl",
        "feature_names.pkl"
    ]
    
    if not os.path.exists(model_dir):
        print(f"⚠ models 目录不存在")
        print("系统将使用默认预测模式")
        return True
    
    all_exist = True
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        if os.path.exists(model_path):
            print(f"✓ {model_file}")
        else:
            print(f"✗ {model_file} (缺失)")
            all_exist = False
    
    if all_exist:
        print("\n✓ 所有模型文件都存在，将使用机器学习模型")
        return True
    else:
        print("\n⚠ 部分模型文件缺失，将使用默认预测模式")
        print("\n默认预测模式说明:")
        print("  - 基于RGB颜色特征的简单线性映射")
        print("  - 无需训练数据即可测试系统功能")
        print("  - 预测准确性较低，仅用于演示和测试")
        print("\n如需使用机器学习模型，请参考 INSTALL.md 中的训练步骤")
        return True


def start_application():
    """启动应用"""
    print_header("启动应用")
    
    try:
        print("正在启动茶色素智能检测系统...")
        print("\n" + "-" * 60)
        
        # 导入主程序
        from main_enhanced import main
        
        # 启动应用
        main()
        
    except ImportError as e:
        print(f"✗ 导入主程序失败: {e}")
        print("\n请确保:")
        print("  1. 所有依赖包已正确安装")
        print("  2. Python路径配置正确")
        return False
        
    except Exception as e:
        print(f"✗ 启动应用失败: {e}")
        print("\n错误详情:")
        import traceback
        traceback.print_exc()
        return False


def show_usage():
    """显示使用说明"""
    print_header("使用说明")
    
    print("""
启动成功后，您可以按照以下步骤使用系统:

1. 打开图片
   - 点击左侧 "打开图片" 按钮
   - 选择茶汤照片 (支持 JPG, PNG 等格式)

2. 设置参数 (可选)
   - 预处理参数: 图像增强、滤波方法、颜色空间
   - 检测参数: 置信度阈值
   - 颜色校准: 需要标准色卡 (可选)

3. 开始检测
   - 点击 "开始检测" 按钮
   - 等待检测完成 (通常 1-3 秒)

4. 查看结果
   - 右侧显示茶色素含量 (TF/TR/TB)
   - 可视化图表展示结果
   - 详细信息表格显示检测元数据

5. 保存结果
   - 点击 "保存结果" 按钮
   - 选择保存路径和格式 (CSV/TXT/JSON)

注意事项:
- 首次使用建议用几张图片测试功能
- 如需高精度检测，请训练机器学习模型
- 拍摄时建议使用标准光源和固定角度
    """)


def main():
    """主函数"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "    茶色素智能检测系统 - 快速启动脚本".center(58) + "║")
    print("║" + "    Tea Pigment Detection System - Quick Start".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    
    # 检查Python版本
    if not check_python_version():
        sys.exit(1)
    
    # 检查虚拟环境
    check_virtual_env()
    
    # 检查项目结构
    if not check_project_structure():
        sys.exit(1)
    
    # 检查模型文件
    check_models()
    
    # 询问是否安装依赖
    print_header("依赖包检查")
    
    response = input("是否检查并安装/更新依赖包? (y/n) [y]: ").strip().lower()
    if response == '' or response == 'y':
        if not install_dependencies():
            print("\n⚠ 依赖包安装可能存在问题，但尝试继续启动...")
    else:
        print("跳过依赖包安装")
    
    # 显示使用说明
    show_usage()
    
    # 询问是否启动
    print_header("准备启动")
    
    response = input("是否立即启动应用? (y/n) [y]: ").strip().lower()
    if response == '' or response == 'y':
        # 切换到脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # 启动应用
        start_application()
    else:
        print("\n启动已取消。您可以稍后运行以下命令启动:")
        print("  python main_enhanced.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已取消操作")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n发生错误: {e}")
        sys.exit(1)