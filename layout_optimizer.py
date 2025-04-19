# 汉语双拼布局优化器 - 使用模拟退火算法寻找最优布局
# 可作为独立程序使用或被fluent_type.py调用

import os
import sys
import json
import random
import math
import numpy as np

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

def random_mapping(keys, fixed_map=None):
    """为给定的键集合生成随机字母映射，同时保持预先固定的映射不变
    
    参数:
    keys - 需要映射的键列表
    fixed_map - 预先固定的映射字典，这些映射不会被改变
    
    返回:
    mapping - 完整的映射字典
    """
    if fixed_map is None:
        fixed_map = {}
    
    # 创建新的映射，首先使用固定映射
    mapping = fixed_map.copy()
    
    # 统一转换为大写以匹配FINGER_MAP
    for k in mapping:
        mapping[k] = mapping[k].upper()
    
    # 获取已经使用的字母
    used_letters = set(mapping.values())
    
    # 可用字母列表 - 只使用在FINGER_MAP中有定义的字母
    available_letters = [letter for letter in ALPHABET if letter in FINGER_MAP and letter not in used_letters]
    
    # 确保有足够的可用字母
    if len(keys) - len(mapping) > len(available_letters):
        print(f"警告：可用字母不足！需要{len(keys) - len(mapping)}个，但只有{len(available_letters)}个可用")
    
    # 对于没有固定映射的键，随机生成映射
    for k in keys:
        if k not in mapping:
            if not available_letters:
                # 如果没有可用字母了，则从FINGER_MAP中选择一个已有的（可能会导致重复）
                mapping[k] = random.choice([key for key in FINGER_MAP.keys() if key in ALPHABET])
                print(f"警告：可用字母不足，为'{k}'分配了已使用的字母'{mapping[k]}'")
            else:
                # 随机选择一个未使用的字母并从可用列表中移除
                letter = random.choice(available_letters)
                mapping[k] = letter
                available_letters.remove(letter)
    
    return mapping

def calc_loss(prob_matrix, initial_map, final_map, finger_map, weight_unfluent=1.0):
    """
    计算布局的损失值
    
    参数:
    prob_matrix - 概率矩阵，表示每个声母韵母组合的概率
    initial_map - 声母到字母的映射
    final_map - 韵母到字母的映射
    finger_map - 字母到手指的映射
    weight_unfluent - 权重因子，用于计算不流畅度的代价
    
    返回:
    loss - 损失值，越小越好
    """
    cost1 = 0.0  # 衡量一根手指连续敲击不同按键的情况
    finger_load = {f: 0.0 for f in COST2_FINGERS}  # 记录每个手指的负载
    
    initials = list(prob_matrix.keys())
    finals = list(prob_matrix[initials[0]].keys())
    
    # 检查映射是否有效
    invalid_keys = []
    for key, letter in initial_map.items():
        if letter not in finger_map:
            invalid_keys.append((key, letter, "声母"))
    for key, letter in final_map.items():
        if letter not in finger_map:
            invalid_keys.append((key, letter, "韵母"))
    
    # 如果有无效映射，打印并返回无穷大
    if invalid_keys:
        print("检测到无效映射（字母不在FINGER_MAP中）:")
        for key, letter, type_str in invalid_keys:
            print(f"  {type_str} '{key}' 映射到 '{letter}', 但'{letter}'不在FINGER_MAP中")
        return float('inf')
    
    # 遍历所有声母韵母组合
    for ini in initials:
        for fin in finals:
            p = prob_matrix[ini][fin]
            if p == 0:
                continue
            
            # 获取映射到的字母
            key1 = initial_map[ini]
            key2 = final_map[fin]
            
            # 获取对应的手指
            f1 = finger_map.get(key1)
            f2 = finger_map.get(key2)
            
            # cost1: 同一个手指连续敲击不同键
            if f1 == f2 and key1 != key2:
                cost1 += p * weight_unfluent
            
            # 手指负载累加
            if f1 in COST2_FINGERS:
                finger_load[f1] += p
            if f2 in COST2_FINGERS:
                finger_load[f2] += p
    
    # 计算手指负载的方差，作为cost2
    loads = [finger_load[f] for f in COST2_FINGERS]
    cost2 = np.var(loads)  # 方差越小，说明负载越均匀
    # print(f"cost1: {cost1:.4f}, cost2: {cost2:.4f}")
    # 总损失 = cost1 + cost2
    return cost1 + 10*cost2

def simulated_annealing(prob_matrix, initials, finals, finger_map, max_iter=200000, fixed_initial_map=None, fixed_final_map=None):
    """
    使用模拟退火算法寻找最优映射，支持预先固定的映射
    
    参数:
    prob_matrix - 概率矩阵
    initials - 声母列表
    finals - 韵母列表
    finger_map - 字母到手指的映射
    max_iter - 最大迭代次数
    fixed_initial_map - 预先固定的声母映射
    fixed_final_map - 预先固定的韵母映射
    
    返回:
    best_map - 最佳映射: (声母映射, 韵母映射)
    best_loss - 最佳损失值
    """
    # 处理传入的固定映射参数
    if fixed_initial_map is None:
        fixed_initial_map = {}
    if fixed_final_map is None:
        fixed_final_map = {}
    
    # 初始化随机映射，保持固定部分不变
    ini_map = random_mapping(initials, fixed_initial_map)
    fin_map = random_mapping(finals, fixed_final_map)
    
    # 计算初始损失
    loss = calc_loss(prob_matrix, ini_map, fin_map, finger_map)
    
    # 记录最佳结果
    best_map = (ini_map.copy(), fin_map.copy())
    best_loss = loss
    
    # 模拟退火参数
    T = 1.0  # 初始温度
    
    # 获取可以变动的声母和韵母（排除固定的）
    variable_initials = [ini for ini in initials if ini not in fixed_initial_map]
    variable_finals = [fin for fin in finals if fin not in fixed_final_map]
    
    # 如果所有的映射都已固定，则直接返回
    if not variable_initials and not variable_finals:
        print("所有映射都已固定，无需优化")
        return best_map, best_loss
    
    print("开始模拟退火算法...")
    print(f"初始损失值: {loss:.4f}")
    print(f"可变声母: {len(variable_initials)}/{len(initials)}")
    print(f"可变韵母: {len(variable_finals)}/{len(finals)}")
    
    # 迭代退火过程
    for it in range(max_iter):
        # 随机选择一个映射进行变动
        if random.random() < 0.5 and variable_initials:
            # 修改声母映射（只选择非固定的）
            k = random.choice(variable_initials)
            new_ini_map = ini_map.copy()
            new_ini_map[k] = random.choice(ALPHABET)
            new_fin_map = fin_map
        elif variable_finals:
            # 修改韵母映射（只选择非固定的）
            k = random.choice(variable_finals)
            new_fin_map = fin_map.copy()
            new_fin_map[k] = random.choice(ALPHABET)
            new_ini_map = ini_map
        else:
            # 如果没有可变更的部分，则跳过
            continue
        
        # 计算新损失
        new_loss = calc_loss(prob_matrix, new_ini_map, new_fin_map, finger_map)
        
        # 计算接受概率
        # 如果新损失更小，则始终接受
        # 否则，根据温度和损失差异计算接受概率
        if new_loss < loss:
            AP = 1.0
        else:
            AP = math.exp((loss - new_loss) / T)
        
        # 根据接受概率决定是否接受新状态
        if AP > random.random():
            ini_map, fin_map, loss = new_ini_map.copy(), new_fin_map.copy(), new_loss
            
            # 如果是目前最佳结果，则更新记录
            if loss < best_loss:
                best_map = (ini_map.copy(), fin_map.copy())
                best_loss = loss
                print(f"迭代 {it+1}/{max_iter}, 温度: {T:.4f}, 找到更优解: {best_loss:.4f}")
        
        # 降低温度
        T *= 0.99999
        
        # 当温度足够低时结束
        if T < 0.01:
            print(f"温度已降至 {T:.4f}，低于阈值 0.01，提前结束退火过程")
            break
        
        # 每1000次迭代打印一次进度
        if (it + 1) % 1000 == 0:
            print(f"迭代进度: {it+1}/{max_iter}, 温度: {T:.4f}, 当前最优损失: {best_loss:.4f}")
    
    return best_map, best_loss

def load_prob_matrix(filepath):
    """从文件加载概率矩阵"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # 第一行是韵母列表
        finals = lines[0].strip().split('\t')[1:]
        
        # 创建概率矩阵
        prob_matrix = {}
        initials = []
        
        # 读取矩阵内容
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) <= 1:
                continue
            
            ini = parts[0]
            initials.append(ini)
            
            # 转换为浮点数
            prob_matrix[ini] = {fin: float(p) for fin, p in zip(finals, parts[1:])}
        
        return initials, finals, prob_matrix
    
    except Exception as e:
        print(f"加载概率矩阵时出错: {str(e)}")
        return None, None, None

def save_mapping(initial_map, final_map, filename="optimal_mapping.json"):
    """保存映射到JSON文件"""
    mapping = {
        "initial_map": initial_map,
        "final_map": final_map
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"保存映射时出错: {str(e)}")
        return False

def load_fixed_mapping(filepath="pre_fixed_mapping.json"):
    """加载预先固定的映射"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
                
                # 获取固定的映射
                fixed_initial_map = mapping.get("initial_map", {})
                fixed_final_map = mapping.get("final_map", {})
                
                # 确保所有字母都是大写(以匹配FINGER_MAP)
                for k in fixed_initial_map:
                    fixed_initial_map[k] = fixed_initial_map[k].upper()
                for k in fixed_final_map:
                    fixed_final_map[k] = fixed_final_map[k].upper()
                
                return fixed_initial_map, fixed_final_map
        
        # 如果文件不存在或出错，返回空映射
        return {}, {}
    except Exception as e:
        print(f"加载预先固定映射时出错: {str(e)}")
        return {}, {}

def analyze_finger_load(prob_matrix, initial_map, final_map):
    """分析手指负载分布"""
    initials = list(prob_matrix.keys())
    finals = list(prob_matrix[initials[0]].keys())
    
    finger_load = {f: 0.0 for f in COST2_FINGERS}
    for ini in initials:
        for fin in finals:
            p = prob_matrix[ini][fin]
            if p == 0:
                continue
            
            key1 = initial_map[ini]
            key2 = final_map[fin]
            
            f1 = FINGER_MAP.get(key1)
            f2 = FINGER_MAP.get(key2)
            
            if f1 in COST2_FINGERS:
                finger_load[f1] += p
            if f2 in COST2_FINGERS:
                finger_load[f2] += p
    
    print("\n手指负载分布:")
    finger_names = {
        1: "左小指", 2: "左无名指", 3: "左中指", 4: "左食指",
        7: "右食指", 8: "右中指", 9: "右无名指", 10: "右小指"
    }
    
    for f in COST2_FINGERS:
        print(f"{finger_names[f]}: {finger_load[f]:.4f}")
    
    # 计算手指负载方差
    loads = [finger_load[f] for f in COST2_FINGERS]
    var = np.var(loads)
    print(f"\n手指负载方差: {var:.6f}")
    
    return finger_load, var

def optimize_layout(prob_matrix_file, fixed_mapping_file="pre_fixed_mapping.json", 
                    mapping_output_file="optimal_mapping.json", max_iter=20000):
    """
    优化布局的主函数，可被外部调用
    
    参数:
    prob_matrix_file - 概率矩阵文件路径
    fixed_mapping_file - 预先固定映射文件路径
    mapping_output_file - 输出映射文件路径
    max_iter - 最大迭代次数
    
    返回:
    success - 是否成功完成优化
    best_map - 最佳映射: (声母映射, 韵母映射)
    best_loss - 最佳损失值
    """
    # 加载概率矩阵
    print(f"正在从文件加载概率矩阵: {prob_matrix_file}")
    initials, finals, prob_matrix = load_prob_matrix(prob_matrix_file)
    if not initials or not finals or not prob_matrix:
        print("无法获取有效的概率矩阵，无法进行双拼布局优化")
        return False, None, None
    
    # 加载预先固定的映射
    fixed_initial_map, fixed_final_map = load_fixed_mapping(fixed_mapping_file)
    if fixed_initial_map or fixed_final_map:
        print("\n检测到预先固定的映射:")
        if fixed_initial_map:
            print(f"固定的声母映射 ({len(fixed_initial_map)} 个):")
            for k, v in sorted(fixed_initial_map.items()):
                print(f"  {k}: {v}")
        if fixed_final_map:
            print(f"固定的韵母映射 ({len(fixed_final_map)} 个):")
            for k, v in sorted(fixed_final_map.items()):
                print(f"  {k}: {v}")
    
    # 检查FINGER_MAP中的键与可以使用的字母
    valid_map_keys = set(FINGER_MAP.keys())
    valid_alphabet = set(ALPHABET) & valid_map_keys
    print(f"\nFINGER_MAP中有效的字母数量: {len(valid_alphabet)}")
    
    # 检查是否有足够的字母可用
    total_mappings_needed = len(initials) + len(finals)
    fixed_mappings_count = len(fixed_initial_map) + len(fixed_final_map)
    remaining_needed = total_mappings_needed - fixed_mappings_count
    
    if len(valid_alphabet) < remaining_needed:
        print(f"警告: 需要分配{remaining_needed}个字母，但只有{len(valid_alphabet)}个有效字母可用")
        print("这可能会导致某些声母/韵母共用同一个字母，影响优化结果")
    
    # 运行模拟退火算法寻找最优映射
    print(f"\n使用模拟退火算法寻找最优双拼布局，最大迭代次数: {max_iter}")
    best_map, best_loss = simulated_annealing(prob_matrix, initials, finals, FINGER_MAP, 
                                             max_iter, fixed_initial_map, fixed_final_map)
    
    print(f"模拟退火算法完成")
    print(f"最优损失值: {best_loss:.4f}")
    
    # 如果最优损失值为无穷大，说明没有找到有效解
    if math.isinf(best_loss):
        print("\n警告: 未能找到有效的映射方案，请检查固定映射和FINGER_MAP的配置")
        return False, best_map, best_loss
    
    # 打印最优映射
    initial_map, final_map = best_map
    print("\n最优声母映射:")
    for k, v in sorted(initial_map.items()):
        print(f"{k}: {v}")
    
    print("\n最优韵母映射:")
    for k, v in sorted(final_map.items()):
        print(f"{k}: {v}")
    
    # 保存最优映射到文件
    if save_mapping(initial_map, final_map, mapping_output_file):
        print(f"\n最优映射已保存到文件: {mapping_output_file}")
    else:
        print("\n保存最优映射失败")
    
    # 分析手指负载分布
    analyze_finger_load(prob_matrix, initial_map, final_map)
    
    return True, best_map, best_loss

def main():
    """可执行程序的入口点"""
    import argparse
    import time
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='汉语双拼布局优化器 - 使用模拟退火算法')
    
    # 添加命令行参数
    parser.add_argument('--matrix', '-m', type=str, default='pinyin_probability_matrix.txt',
                        help='概率矩阵文件路径 (默认: pinyin_probability_matrix.txt)')
    parser.add_argument('--fixed', '-f', type=str, default='pre_fixed_mapping.json',
                        help='预先固定映射文件路径 (默认: pre_fixed_mapping.json)')
    parser.add_argument('--output', '-o', type=str, default='optimal_mapping.json',
                        help='最优映射输出文件路径 (默认: optimal_mapping.json)')
    parser.add_argument('--iter', '-i', type=int, default=20000,
                        help='最大迭代次数 (默认: 20000)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    print("="*50)
    print("汉语双拼布局优化器")
    print("="*50)
    print(f"概率矩阵文件: {args.matrix}")
    print(f"预先固定映射文件: {args.fixed}")
    print(f"输出映射文件: {args.output}")
    print(f"最大迭代次数: {args.iter}")
    print("="*50)
    
    # 记录开始时间
    start_time = time.time()
    
    # 执行优化
    success, _, _ = optimize_layout(args.matrix, args.fixed, args.output, args.iter)
    
    # 输出总用时
    if success:
        total_time = time.time() - start_time
        print(f"\n总计用时: {total_time:.2f} 秒")

if __name__ == "__main__":
    main()

'''
python layout_optimizer.py --matrix pinyin_probability_matrix.txt --fixed pre_fixed_mapping.json --output optimal_mapping.json --iter 20000
'''
