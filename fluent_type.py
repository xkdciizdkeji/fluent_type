# 汉语双拼布局优化工具
# 作者：GitHub Copilot

import os
import sys
import subprocess

def run_pinyin_generator(input_file=None, output_file="pinyin_string.txt"):
    """
    运行拼音生成器
    
    Args:
        input_file: 输入的语料库文件路径
        output_file: 输出的拼音串文件路径
    
    Returns:
        成功与否
    """
    print("正在运行拼音生成器...")
    
    cmd = ["python", "pinyin_generator.py"]
    if input_file:
        cmd.append(input_file)
    if output_file:
        cmd.append(output_file)
    
    try:
        process = subprocess.run(cmd, check=True)
        return process.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"拼音生成器运行失败: {e}")
        return False
    
def run_matrix_generator(input_file="pinyin_string.txt", matrix_file="pinyin_frequency_matrix.txt", image_file="pinyin_frequency_matrix.png"):
    """
    运行矩阵生成器
    
    Args:
        input_file: 输入的拼音串文件路径
        matrix_file: 输出的矩阵文件路径
        image_file: 输出的图像文件路径
    
    Returns:
        成功与否
    """
    print("正在运行矩阵生成器...")
    
    cmd = ["python", "matrix_generator.py", input_file, matrix_file, image_file]
    
    try:
        process = subprocess.run(cmd, check=True)
        return process.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"矩阵生成器运行失败: {e}")
        return False

def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 1:
        # 如果提供了文件路径，作为语料库文件路径
        corpus_file = sys.argv[1]
        pinyin_file = sys.argv[2] if len(sys.argv) > 2 else "pinyin_string.txt"
        matrix_file = sys.argv[3] if len(sys.argv) > 3 else "pinyin_frequency_matrix.txt"
        image_file = sys.argv[4] if len(sys.argv) > 4 else "pinyin_frequency_matrix.png"
        
        print(f"语料库文件: {corpus_file}")
        print(f"拼音串文件: {pinyin_file}")
        print(f"矩阵文件: {matrix_file}")
        print(f"热力图文件: {image_file}")
        
        # 步骤1: 生成拼音串
        if run_pinyin_generator(corpus_file, pinyin_file):
            print("拼音串生成成功!")
            # 步骤2: 生成频率矩阵
            if run_matrix_generator(pinyin_file, matrix_file, image_file):
                print("频率矩阵生成成功!")
                print(f"请查看生成的热力图: {image_file}")
            else:
                print("频率矩阵生成失败")
        else:
            print("拼音串生成失败")
    else:
        # 如果没有提供文件路径，运行测试流程
        print("未提供语料库文件路径，运行测试...")
        # 步骤1: 生成拼音串
        if run_pinyin_generator():
            print("拼音串生成成功!")
            # 步骤2: 生成频率矩阵
            if run_matrix_generator():
                print("频率矩阵生成成功!")
                print("请查看生成的热力图: pinyin_frequency_matrix.png")
            else:
                print("频率矩阵生成失败")
        else:
            print("拼音串生成失败")

if __name__ == "__main__":
    main()