
import os
import time
import json
import multiprocessing
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np
from collections import defaultdict

# 导入拼音生成器和矩阵生成器模块
import pinyin_generator as pg

def process_file(file_path, max_chars=None):
    """
    处理单个文件，生成拼音串
    
    Args:
        file_path: 文件路径
        max_chars: 最大处理字符数
    
    Returns:
        拼音串
    """
    try:
        # 加载文件内容
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # 限制字符数量
            if max_chars and len(content) > max_chars:
                content = content[:max_chars]
                
        # 生成拼音串
        pinyin_string = pg.get_pinyin_string(content)
        return pinyin_string
    except Exception as e:
        print(f"处理文件 {file_path} 出错: {e}")
        return []

def collect_files(folder_path, max_files=None):
    """
    收集文件夹中的所有文件路径
    
    Args:
        folder_path: 文件夹路径
        max_files: 最大文件数量限制
    
    Returns:
        文件路径列表
    """
    all_files = []
    
    # 遍历所有子文件夹和文件
    for root, _, files in os.walk(folder_path):
        for file in files:
            # 跳过隐藏文件和临时文件
            if file.startswith('.') or file.endswith('~'):
                continue
                
            file_path = os.path.join(root, file)
            all_files.append(file_path)
            
            # 如果达到最大文件数量限制，则返回
            if max_files and len(all_files) >= max_files:
                return all_files
    
    return all_files

def merge_pinyin_strings(pinyin_strings_list):
    """
    合并多个拼音串列表
    
    Args:
        pinyin_strings_list: 多个拼音串列表
    
    Returns:
        合并后的拼音串列表
    """
    merged = []
    for pinyin_strings in pinyin_strings_list:
        if pinyin_strings:
            merged.extend(pinyin_strings)
    return merged

def process_corpus_parallel(folder_path, max_files=None, max_chars_per_file=None, num_processes=None):
    """
    并行处理语料库文件夹
    
    Args:
        folder_path: 文件夹路径
        max_files: 最大处理文件数量
        max_chars_per_file: 每个文件最大处理字符数
        num_processes: 并行处理的进程数，默认为CPU核心数
    
    Returns:
        合并后的拼音串列表
    """
    # 如果num_processes为None，使用CPU核心数量
    if num_processes is None:
        num_processes = cpu_count()
        
    print(f"使用 {num_processes} 个进程进行并行处理")
    
    start_time = time.time()
    
    # 收集文件路径
    print("正在收集文件路径...")
    file_paths = collect_files(folder_path, max_files)
    total_files = len(file_paths)
    print(f"找到 {total_files} 个文件")
    
    # 创建进程池
    with Pool(processes=num_processes) as pool:
        # 使用partial函数固定max_chars参数
        process_file_with_max_chars = partial(process_file, max_chars=max_chars_per_file)
        
        # 使用imap处理文件，每处理10个文件显示一次进度
        results = []
        for i, result in enumerate(pool.imap(process_file_with_max_chars, file_paths)):
            results.append(result)
            if (i + 1) % 10 == 0 or (i + 1) == total_files:
                progress = (i + 1) / total_files * 100
                elapsed_time = time.time() - start_time
                estimated_total = elapsed_time / (i + 1) * total_files
                remaining_time = estimated_total - elapsed_time
                print(f"处理进度: {i+1}/{total_files} ({progress:.1f}%), "
                      f"已用时: {elapsed_time:.1f}s, "
                      f"预计剩余时间: {remaining_time:.1f}s")
    
    # 合并所有拼音串列表
    print("正在合并拼音串...")
    merged_pinyin_strings = merge_pinyin_strings(results)
    
    total_time = time.time() - start_time
    print(f"并行处理完成，总用时: {total_time:.1f}秒")
    
    return merged_pinyin_strings

def count_initial_final_pairs(pinyin_strings):
    """
    统计声母-韵母对的出现次数
    
    Args:
        pinyin_strings: 拼音串列表
    
    Returns:
        initial_final_counts: 声母-韵母对出现次数字典
        initials_set: 声母集合
        finals_set: 韵母集合
    """
    # 统计声母-韵母对的出现次数
    initial_final_counts = defaultdict(int)
    
    # 所有可能的声母和韵母集合
    initials_set = set()
    finals_set = set()
    
    # 遍历所有拼音字符串
    for sentence in pinyin_strings:
        for pair in sentence:
            initial, final = pair
            initials_set.add(initial)
            finals_set.add(final)
            initial_final_counts[(initial, final)] += 1
    
    return initial_final_counts, initials_set, finals_set

def create_matrix_from_counts(initial_final_counts, initials, finals):
    """
    从计数字典创建频率矩阵
    
    Args:
        initial_final_counts: 声母-韵母对出现次数字典
        initials: 声母列表
        finals: 韵母列表
    
    Returns:
        matrix: 频率矩阵
    """
    # 创建频率矩阵
    matrix = np.zeros((len(initials), len(finals)))
    
    # 填充矩阵
    for i, initial in enumerate(initials):
        for j, final in enumerate(finals):
            matrix[i, j] = initial_final_counts.get((initial, final), 0)
    
    return matrix

def process_folder_to_pinyin_string(folder_path, output_file="pinyin_string.txt", 
                                   max_files=None, max_chars_per_file=None, num_processes=None):
    """
    并行处理文件夹到拼音串并保存
    
    Args:
        folder_path: 文件夹路径
        output_file: 输出文件路径
        max_files: 最大处理文件数量
        max_chars_per_file: 每个文件最大处理字符数
        num_processes: 并行处理的进程数
    
    Returns:
        成功返回True，失败返回False
    """
    # 并行处理语料库
    pinyin_strings = process_corpus_parallel(
        folder_path, max_files, max_chars_per_file, num_processes
    )
    
    # 保存拼音串
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pinyin_strings, f, ensure_ascii=False, indent=2)
        print(f"拼音串已保存到 {output_file}")
        return True, pinyin_strings
    except Exception as e:
        print(f"保存拼音串失败: {e}")
        return False, pinyin_strings