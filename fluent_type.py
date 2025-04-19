# 汉语双拼布局优化工具
# 作者：GitHub Copilot

import os
import sys
import json
import time
import random
import math
import numpy as np

# 导入拼音生成器、矩阵生成器和并行处理模块
import pinyin_generator as pg
import matrix_generator as mg
import parallel_processor as pp

# 导入布局优化器
import layout_optimizer as lo

# 定义26键键盘的标准指法映射
# 手指编号：左手小指=1, 左手无名指=2, 左手中指=3, 左手食指=4, 左手拇指=5, 
#         右手拇指=6, 右手食指=7, 右手中指=8, 右手无名指=9, 右手小指=10
FINGER_MAP = {
    'Q': 1, 'A': 1, 'Z': 1,
    'W': 2, 'S': 2, 'X': 2,
    'E': 3, 'D': 3, 'C': 3,
    'R': 4, 'F': 4, 'V': 4, 'T': 4, 'G': 4, 'B': 4,
    'Y': 7, 'H': 7, 'N': 7, 'U': 7, 'J': 7, 'M': 7,
    'I': 8, 'K': 8, ',': 8,
    'O': 9, 'L': 9, '.': 9,
    'P': 10, ';': 10, '/': 10, '[': 10, ']': 10, '\\': 10
}

# 用于损失函数计算的手指列表（不包括拇指）
COST2_FINGERS = [1, 2, 3, 4, 7, 8, 9, 10]

# 字母表
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def main():
    """主函数"""
    # 设置wiki_zh路径
    wiki_zh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wiki_zh")
    
    # 设置其他默认参数
    pinyin_file = "pinyin_string.txt"
    freq_matrix_file = "pinyin_frequency_matrix.txt"
    freq_image_file = "pinyin_frequency_matrix.png"
    prob_matrix_file = "pinyin_probability_matrix.txt"
    prob_image_file = "pinyin_probability_matrix.png"
    mapping_file = "optimal_mapping.json"
    
    # 并行处理的参数
    max_files = 200        # 默认处理的最大文件数
    max_chars_per_file = 50000  # 每个文件最多处理的字符数
    num_processes = None   # 默认使用所有可用CPU核心
    
    # 模拟退火参数
    max_iter = 20000       # 最大迭代次数
    run_sa = True         # 是否运行模拟退火算法
    
    # 处理命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print("用法: python fluent_type.py [输入路径] [拼音文件] [频率矩阵文件] [频率图片] [概率矩阵文件] [概率图片] [最大文件数] [每文件最大字符数] [进程数]")
            print("示例: python fluent_type.py wiki_zh pinyin_string.txt pinyin_frequency_matrix.txt pinyin_frequency_matrix.png pinyin_probability_matrix.txt pinyin_probability_matrix.png 200 50000 8")
            return
        
        # 如果提供了路径，使用它代替默认的wiki_zh路径
        input_path = sys.argv[1]
        
        # 处理可选参数
        if len(sys.argv) > 2:
            pinyin_file = sys.argv[2]
        if len(sys.argv) > 3:
            freq_matrix_file = sys.argv[3]
        if len(sys.argv) > 4:
            freq_image_file = sys.argv[4]
        if len(sys.argv) > 5:
            prob_matrix_file = sys.argv[5]
        if len(sys.argv) > 6:
            prob_image_file = sys.argv[6]
        if len(sys.argv) > 7:
            max_files = int(sys.argv[7])
        if len(sys.argv) > 8:
            max_chars_per_file = int(sys.argv[8])
        if len(sys.argv) > 9:
            num_processes = int(sys.argv[9])
    else:
        # 使用默认wiki_zh路径
        input_path = wiki_zh_path
        print(f"未提供输入路径，使用默认wiki_zh路径: {wiki_zh_path}")
    
    # 打印配置信息
    print(f"输入路径: {input_path}")
    print(f"拼音串文件: {pinyin_file}")
    print(f"频率矩阵文件: {freq_matrix_file}")
    print(f"频率热力图文件: {freq_image_file}")
    print(f"概率矩阵文件: {prob_matrix_file}")
    print(f"概率热力图文件: {prob_image_file}")
    print(f"最优映射文件: {mapping_file}")
    print(f"最大文件数: {max_files}")
    print(f"每个文件最大字符数: {max_chars_per_file}")
    if num_processes:
        print(f"并行进程数: {num_processes}")
    else:
        print(f"并行进程数: 自动(使用所有CPU核心)")
    
    # 记录开始时间
    start_time = time.time()
    
    # 检查是路径是文件还是文件夹
    if os.path.isdir(input_path):
        print(f"检测到文件夹 {input_path}，将使用并行处理...")
        
        # 步骤1: 使用并行处理生成拼音串
        success, pinyin_string = pp.process_folder_to_pinyin_string(
            input_path, pinyin_file, max_files, max_chars_per_file, num_processes
        )
        
        if not success:
            print("处理失败，程序终止")
            return
    else:
        print(f"检测到单个文件 {input_path}，直接处理...")
        # 使用原来的单进程方式处理单个文件
        corpus_text = pg.load_corpus(input_path)
        if not corpus_text:
            print("语料库为空，请检查文件路径")
            return
            
        load_time = time.time()
        print(f"成功加载语料库，共 {len(corpus_text)} 个字符，用时 {load_time - start_time:.2f} 秒")
        
        # 生成拼音串
        print("正在生成拼音串...")
        pinyin_string = pg.get_pinyin_string(corpus_text)
        pinyin_time = time.time()
        print(f"拼音串生成完成，用时 {pinyin_time - load_time:.2f} 秒")
        
        # 保存拼音串
        if not pg.save_pinyin_string(pinyin_string, pinyin_file):
            print("保存拼音串失败，程序终止")
            return
    
    # 步骤2: 生成频率矩阵和概率矩阵
    print("正在生成频率矩阵和概率矩阵...")
    matrix_start_time = time.time()
    
    # 创建频率矩阵
    initials, finals, freq_matrix = mg.create_frequency_matrix(pinyin_string)
    
    # 创建概率矩阵
    prob_matrix, total_count = mg.create_probability_matrix(initials, finals, freq_matrix)
    
    matrix_time = time.time()
    print(f"矩阵生成完成，用时 {matrix_time - matrix_start_time:.2f} 秒")
    print(f"发现 {len(initials)} 个声母, {len(finals)} 个韵母, 总计 {int(total_count)} 个拼音")
    print(f"广义声母: {initials}")
    print(f"广义韵母: {finals}")
    
    # 保存频率矩阵到文件
    if mg.save_matrix_csv(initials, finals, freq_matrix, freq_matrix_file):
        print(f"频率矩阵已保存到文件: {freq_matrix_file}")
    else:
        print("保存频率矩阵失败")
    
    # 保存概率矩阵到文件
    if mg.save_probability_matrix_csv(initials, finals, prob_matrix, prob_matrix_file):
        print(f"概率矩阵已保存到文件: {prob_matrix_file}")
    else:
        print("保存概率矩阵失败")
    
    # 生成频率矩阵和概率矩阵的可视化
    print("正在生成可视化...")
    viz_start_time = time.time()
    
    # 频率矩阵可视化
    mg.visualize_frequency_matrix(initials, finals, freq_matrix, "拼音声母韵母频率矩阵", freq_image_file)
    
    # 概率矩阵可视化
    mg.visualize_probability_matrix(initials, finals, prob_matrix, "拼音声母韵母概率矩阵", prob_image_file)
    
    viz_time = time.time()
    print(f"可视化完成，用时 {viz_time - viz_start_time:.2f} 秒")
    
    # 步骤3: 使用模拟退火算法优化双拼布局
    if run_sa:
        print("\n" + "="*50)
        print("开始优化双拼布局...")
        print("="*50)
        
        # 调用布局优化器
        sa_start_time = time.time()
        
        # 使用布局优化器来优化布局
        fixed_mapping_file = "pre_fixed_mapping.json"
        success, best_map, best_loss = lo.optimize_layout(
            prob_matrix_file, fixed_mapping_file, mapping_file, max_iter
        )
        
        sa_time = time.time()
        print(f"布局优化完成，用时 {sa_time - sa_start_time:.2f} 秒")
    
    # 汇总总用时
    total_time = time.time() - start_time
    print(f"\n总计用时: {total_time:.2f} 秒")

if __name__ == "__main__":
    # 在Windows上，确保multiprocessing正确启动子进程
    if os.name == 'nt':  # Windows系统
        from multiprocessing import freeze_support
        freeze_support()
    main()