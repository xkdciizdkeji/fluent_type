# 双拼布局对比测试台
# 用于生成各种对比图，评估不同输入法方案的性能

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

#显示图的中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体


# 导入布局优化器中的计算损失函数
from layout_optimizer import (
    load_prob_matrix, calc_loss, FINGER_MAP, COST2_FINGERS, analyze_finger_load
)

# 定义市面上常见的几种双拼方案
COMMON_SHUANGPIN_SCHEMES = {
    # 小鹤双拼
    "小鹤双拼": {
        "initial_map": {
            "b": "B", "c": "C", "d": "D", "f": "F", "g": "G", "h": "H", "j": "J",
            "k": "K", "l": "L", "m": "M", "n": "N", "p": "P", "q": "Q", "r": "R",
            "s": "S", "t": "T", "w": "W", "x": "X", "y": "Y", "z": "Z",
            "ch": "I", "sh": "U", "zh": "V"
        },
        "final_map": {
            "a": "A", "e": "E", "i": "I", "o": "O", "u": "U", "v": "V",
            "ai": "D", "an": "J", "ang": "H", "ao": "C", "ei": "W", "en": "F",
            "eng": "G", "er": "R", "ia": "X", "ian": "M", "iang": "L", "iao": "N",
            "ie": "P", "in": "B", "ing": "K", "iong": "S", "iu": "Q", "ong": "S",
            "ou": "Z", "ua": "X", "uai": "K", "uan": "R", "uang": "L", "ue": "T",
            "ui": "V", "un": "Y", "uo": "O"
        }
    },
    
    # 微软双拼
    "微软双拼": {
        "initial_map": {
            "b": "B", "c": "C", "d": "D", "f": "F", "g": "G", "h": "H", "j": "J",
            "k": "K", "l": "L", "m": "M", "n": "N", "p": "P", "q": "Q", "r": "R",
            "s": "S", "t": "T", "w": "W", "x": "X", "y": "Y", "z": "Z",
            "ch": "I", "sh": "U", "zh": "V"
        },
        "final_map": {
            "a": "A", "e": "E", "i": "I", "o": "O", "u": "U", "v": "V",
            "ai": "L", "an": "J", "ang": "H", "ao": "K", "ei": "Z", "en": "F",
            "eng": "G", "er": "R", "ia": "W", "ian": "M", "iang": "D", "iao": "C",
            "ie": "X", "in": "N", "ing": ";", "iong": "S", "iu": "Q", "ong": "S",
            "ou": "B", "ua": "W", "uai": "Y", "uan": "R", "uang": "D", "ue": "T",
            "ui": "V", "un": "P", "uo": "O"
        }
    },
    
    # 搜狗双拼
    "搜狗双拼": {
        "initial_map": {
            "b": "B", "c": "C", "d": "D", "f": "F", "g": "G", "h": "H", "j": "J",
            "k": "K", "l": "L", "m": "M", "n": "N", "p": "P", "q": "Q", "r": "R",
            "s": "S", "t": "T", "w": "W", "x": "X", "y": "Y", "z": "Z",
            "ch": "I", "sh": "U", "zh": "V"
        },
        "final_map": {
            "a": "A", "e": "E", "i": "I", "o": "O", "u": "U", "v": "V",
            "ai": "L", "an": "J", "ang": "H", "ao": "K", "ei": "Z", "en": "F",
            "eng": "G", "er": "R", "ia": "W", "ian": "M", "iang": "D", "iao": "C",
            "ie": "X", "in": "N", "ing": ";", "iong": "S", "iu": "Q", "ong": "S",
            "ou": "B", "ua": "W", "uai": "Y", "uan": "R", "uang": "D", "ue": "T",
            "ui": "V", "un": "P", "uo": "O"
        }
    },
    
    # 自然码双拼
    "自然码双拼": {
        "initial_map": {
            "b": "B", "c": "C", "d": "D", "f": "F", "g": "G", "h": "H", "j": "J",
            "k": "K", "l": "L", "m": "M", "n": "N", "p": "P", "q": "Q", "r": "R",
            "s": "S", "t": "T", "w": "W", "x": "X", "y": "Y", "z": "Z",
            "ch": "I", "sh": "U", "zh": "V"
        },
        "final_map": {
            "a": "A", "e": "E", "i": "I", "o": "O", "u": "U", "v": "V",
            "ai": "L", "an": "J", "ang": "H", "ao": "K", "ei": "Z", "en": "F",
            "eng": "G", "er": "R", "ia": "B", "ian": "J", "iang": "D", "iao": "C",
            "ie": "X", "in": "N", "ing": "Y", "iong": "S", "iu": "Q", "ong": "S",
            "ou": "B", "ua": "B", "uai": "Y", "uan": "R", "uang": "D", "ue": "T",
            "ui": "V", "un": "P", "uo": "O"
        }
    },
    
    # 紫光双拼
    "紫光双拼": {
        "initial_map": {
            "b": "B", "c": "C", "d": "D", "f": "F", "g": "G", "h": "H", "j": "J",
            "k": "K", "l": "L", "m": "M", "n": "N", "p": "P", "q": "Q", "r": "R",
            "s": "S", "t": "T", "w": "W", "x": "X", "y": "Y", "z": "Z",
            "ch": "I", "sh": "U", "zh": "V"
        },
        "final_map": {
            "a": "A", "e": "E", "i": "I", "o": "O", "u": "U", "v": "V",
            "ai": "L", "an": "J", "ang": "H", "ao": "K", "ei": "Z", "en": "F",
            "eng": "G", "er": "R", "ia": "W", "ian": "M", "iang": "D", "iao": "C",
            "ie": "X", "in": "N", "ing": ";", "iong": "S", "iu": "Q", "ong": "S",
            "ou": "B", "ua": "W", "uai": "Y", "uan": "R", "uang": "D", "ue": "T",
            "ve": "T", "ui": "V", "un": "P", "uo": "O"
        }
    }
}

def load_optimal_mapping(filename="optimal_mapping.json"):
    """加载优化后的双拼映射"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
            # 确保所有字母都是大写
            for k in mapping["initial_map"]:
                mapping["initial_map"][k] = mapping["initial_map"][k].upper()
            for k in mapping["final_map"]:
                mapping["final_map"][k] = mapping["final_map"][k].upper()
            return mapping
    except Exception as e:
        print(f"加载优化映射时出错: {str(e)}")
        return None

def calculate_keystrokes(prob_matrix, scheme=None):
    """
    计算输入方案的击键次数
    
    参数:
    prob_matrix - 概率矩阵
    scheme - 双拼方案的映射字典(包含initial_map和final_map)
             如果为None，则表示全拼输入法
    
    返回:
    total_keystrokes - 总击键次数
    """
    total_keystrokes = 0
    
    initials = list(prob_matrix.keys())
    finals = list(prob_matrix[initials[0]].keys())
    
    # 遍历所有声母韵母组合
    for ini in initials:
        for fin in finals:
            p = prob_matrix[ini][fin]
            if p == 0:
                continue
            
            # 全拼输入法的击键次数
            if scheme is None:
                # 声母的击键次数
                ini_strokes = len(ini) if ini != "' '" else 0
                # 韵母的击键次数
                fin_strokes = len(fin) if fin != "' '" else 0
                total_keystrokes += p * (ini_strokes + fin_strokes)
            else:
                # 双拼输入法固定为2次击键
                total_keystrokes += p * 2
    
    return total_keystrokes

def calculate_fluency_cost(prob_matrix, scheme, finger_map):
    """
    计算输入方案的流畅度代价
    
    参数:
    prob_matrix - 概率矩阵
    scheme - 双拼方案的映射字典
    finger_map - 字母到手指的映射
    
    返回:
    fluency_cost - 流畅度代价，越小表示越流畅
    """
    cost = 0.0
    
    initials = list(prob_matrix.keys())
    finals = list(prob_matrix[initials[0]].keys())
    
    initial_map = scheme["initial_map"]
    final_map = scheme["final_map"]
    
    # 遍历所有声母韵母组合
    for ini in initials:
        for fin in finals:
            p = prob_matrix[ini][fin]
            if p == 0:
                continue
            
            # 获取映射到的字母
            key1 = initial_map.get(ini, "")
            key2 = final_map.get(fin, "")
            
            # 如果声母或韵母不在映射表中，跳过
            if not key1 or not key2:
                continue
                
            # 获取对应的手指
            f1 = finger_map.get(key1)
            f2 = finger_map.get(key2)
            
            # 如果字母不在指法映射中，跳过
            if f1 is None or f2 is None:
                continue
            
            # 同一个手指连续敲击不同键的情况
            if f1 == f2 and key1 != key2:
                cost += p
    
    return cost

def calculate_load_balance(prob_matrix, scheme, finger_map):
    """
    计算输入方案的手指负载均衡度
    
    参数:
    prob_matrix - 概率矩阵
    scheme - 双拼方案的映射字典
    finger_map - 字母到手指的映射
    
    返回:
    load_variance - 手指负载方差，越小表示负载越均匀
    """
    finger_load = {f: 0.0 for f in COST2_FINGERS}
    
    initials = list(prob_matrix.keys())
    finals = list(prob_matrix[initials[0]].keys())
    
    initial_map = scheme["initial_map"]
    final_map = scheme["final_map"]
    
    # 遍历所有声母韵母组合
    for ini in initials:
        for fin in finals:
            p = prob_matrix[ini][fin]
            if p == 0:
                continue
            
            # 获取映射到的字母
            key1 = initial_map.get(ini, "")
            key2 = final_map.get(fin, "")
            
            # 如果声母或韵母不在映射表中，跳过
            if not key1 or not key2:
                continue
                
            # 获取对应的手指
            f1 = finger_map.get(key1)
            f2 = finger_map.get(key2)
            
            # 如果字母不在指法映射中，跳过
            if f1 is None or f2 is None:
                continue
            
            # 手指负载累加
            if f1 in COST2_FINGERS:
                finger_load[f1] += p
            if f2 in COST2_FINGERS:
                finger_load[f2] += p
    
    # 计算手指负载方差
    loads = [finger_load[f] for f in COST2_FINGERS]
    return np.var(loads)

def draw_keystroke_comparison(prob_matrix_file, optimal_mapping_file):
    """
    绘制击键次数对比图(全拼vs双拼)
    
    参数:
    prob_matrix_file - 概率矩阵文件路径
    optimal_mapping_file - 优化双拼映射文件路径
    """
    # 加载概率矩阵
    initials, finals, prob_matrix = load_prob_matrix(prob_matrix_file)
    if not prob_matrix:
        print("无法加载概率矩阵")
        return
    
    # 加载优化双拼映射
    optimal_mapping = load_optimal_mapping(optimal_mapping_file)
    if not optimal_mapping:
        print("无法加载优化双拼映射")
        return
    
    # 计算全拼和双拼的击键次数
    full_pinyin_keystrokes = calculate_keystrokes(prob_matrix)
    optimal_shuangpin_keystrokes = calculate_keystrokes(prob_matrix, optimal_mapping)
    
    # 计算双拼相对全拼的击键比例
    ratio = (optimal_shuangpin_keystrokes / full_pinyin_keystrokes) * 100
    
    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(['全拼输入法', '本双拼布局'], [100, ratio], color=['#3498db', '#e74c3c'])
    
    # 在柱状上方显示实际百分比
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom')
    
    plt.ylim(0, 110)  # 设置y轴范围
    plt.ylabel('击键次数相对全拼的百分比 (%)')
    plt.title('全拼输入法vs本双拼布局的击键次数对比')
    plt.grid(axis='y', alpha=0.3)
    
    # 保存图像
    plt.savefig('keystroke_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"击键次数对比图已保存为 'keystroke_comparison.png'")
    print(f"全拼输入法击键次数: {full_pinyin_keystrokes:.4f}")
    print(f"本双拼布局击键次数: {optimal_shuangpin_keystrokes:.4f}")
    print(f"击键次数比例: {ratio:.1f}%")

def draw_fluency_comparison(prob_matrix_file, optimal_mapping_file):
    """
    绘制击键流畅性对比图(本双拼布局vs其他双拼方案)
    
    参数:
    prob_matrix_file - 概率矩阵文件路径
    optimal_mapping_file - 优化双拼映射文件路径
    """
    # 加载概率矩阵
    initials, finals, prob_matrix = load_prob_matrix(prob_matrix_file)
    if not prob_matrix:
        print("无法加载概率矩阵")
        return
    
    # 加载优化双拼映射
    optimal_mapping = load_optimal_mapping(optimal_mapping_file)
    if not optimal_mapping:
        print("无法加载优化双拼映射")
        return
    
    # 计算各方案的流畅度代价
    fluency_costs = {}
    
    # 计算本双拼布局的流畅度代价
    fluency_costs["本双拼布局"] = calculate_fluency_cost(prob_matrix, optimal_mapping, FINGER_MAP)
    
    # 计算其他方案的流畅度代价
    for name, scheme in COMMON_SHUANGPIN_SCHEMES.items():
        fluency_costs[name] = calculate_fluency_cost(prob_matrix, scheme, FINGER_MAP)
    
    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    
    # 排序后绘制
    sorted_items = sorted(fluency_costs.items(), key=lambda x: x[1])
    schemes = [item[0] for item in sorted_items]
    costs = [item[1] for item in sorted_items]
    
    # 为本双拼布局使用不同颜色
    colors = ['#e74c3c' if scheme == "本双拼布局" else '#3498db' for scheme in schemes]
    
    bars = plt.bar(schemes, costs, color=colors)
    
    # 在柱状上方显示实际值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')
    
    plt.ylabel('流畅度代价 (越低越好)')
    plt.title('不同双拼方案的击键流畅性对比')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('fluency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"击键流畅性对比图已保存为 'fluency_comparison.png'")
    for name, cost in sorted_items:
        print(f"{name} 流畅度代价: {cost:.4f}")

def draw_load_balance_comparison(prob_matrix_file, optimal_mapping_file):
    """
    绘制手指负载均衡度对比图(本双拼布局vs其他双拼方案)
    
    参数:
    prob_matrix_file - 概率矩阵文件路径
    optimal_mapping_file - 优化双拼映射文件路径
    """
    # 加载概率矩阵
    initials, finals, prob_matrix = load_prob_matrix(prob_matrix_file)
    if not prob_matrix:
        print("无法加载概率矩阵")
        return
    
    # 加载优化双拼映射
    optimal_mapping = load_optimal_mapping(optimal_mapping_file)
    if not optimal_mapping:
        print("无法加载优化双拼映射")
        return
    
    # 计算各方案的负载均衡度
    load_variances = {}
    
    # 计算本双拼布局的负载均衡度
    load_variances["本双拼布局"] = calculate_load_balance(prob_matrix, optimal_mapping, FINGER_MAP)
    
    # 计算其他方案的负载均衡度
    for name, scheme in COMMON_SHUANGPIN_SCHEMES.items():
        load_variances[name] = calculate_load_balance(prob_matrix, scheme, FINGER_MAP)
    
    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    
    # 排序后绘制
    sorted_items = sorted(load_variances.items(), key=lambda x: x[1])
    schemes = [item[0] for item in sorted_items]
    variances = [item[1] for item in sorted_items]
    
    # 为本双拼布局使用不同颜色
    colors = ['#e74c3c' if scheme == "本双拼布局" else '#3498db' for scheme in schemes]
    
    bars = plt.bar(schemes, variances, color=colors)
    
    # 在柱状上方显示实际值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.6f}',
                 ha='center', va='bottom')
    
    plt.ylabel('手指负载方差 (越低越均衡)')
    plt.title('不同双拼方案的手指负载均衡度对比')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('load_balance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"手指负载均衡度对比图已保存为 'load_balance_comparison.png'")
    for name, variance in sorted_items:
        print(f"{name} 手指负载方差: {variance:.6f}")

def draw_keyboard_layout(optimal_mapping_file):
    """
    绘制键盘键位图
    
    参数:
    optimal_mapping_file - 优化双拼映射文件路径
    """
    # 加载优化双拼映射
    mapping = load_optimal_mapping(optimal_mapping_file)
    if not mapping:
        print("无法加载优化双拼映射")
        return
    
    # 反转映射，获取每个键位上对应的声母和韵母
    key_to_initial = {}
    key_to_final = {}
    
    for ini, key in mapping["initial_map"].items():
        if key not in key_to_initial:
            key_to_initial[key] = []
        key_to_initial[key].append(ini)
    
    for fin, key in mapping["final_map"].items():
        if key not in key_to_final:
            key_to_final[key] = []
        key_to_final[key].append(fin)
    
    # 定义键盘布局 - 注意顺序从上到下
    keyboard_layout = [
        ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ],
        ["Z", "X", "C", "V", "B", "N", "M",]
    ]
    
    # 设置图像尺寸
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # 设置背景色
    ax.set_facecolor('#f0f0f0')
    
    # 键盘尺寸参数
    key_width = 1.0
    key_height = 1.0
    key_spacing = 0.2
    
    # 设置颜色
    main_color = '#333333'  # 主要文字颜色
    ini_color = '#009900'   # 声母颜色
    fin_color = '#0066cc'   # 韵母颜色
    
    # 绘制键盘 - 从上到下绘制，确保正确的顺序显示
    total_rows = len(keyboard_layout)
    for row_idx, row in enumerate(keyboard_layout):
        # 计算该行的偏移量，使键盘居中
        y_position = (total_rows - row_idx - 1) * (key_height + key_spacing)  # 从上到下布局
        row_offset = row_idx * 0.5
        
        for col_idx, key in enumerate(row):
            x = col_idx * (key_width + key_spacing) + row_offset
            y = y_position  # 使用计算好的y位置
            
            # 创建键位的矩形（使用FancyBboxPatch支持圆角）
            rect = patches.FancyBboxPatch(
                (x, y), key_width, key_height, 
                linewidth=1, edgecolor='gray', facecolor='white',
                boxstyle='round,pad=0.1', zorder=1
            )
            
            # 添加阴影效果
            shadow = patches.Rectangle(
                (x + 0.03, y + 0.03), key_width, key_height, 
                linewidth=0, facecolor='gray', alpha=0.2, zorder=0
            )
            
            # 添加键位矩形和阴影到图中
            ax.add_patch(shadow)
            ax.add_patch(rect)
            
            # 在键位中添加字母标签
            ax.text(x + key_width/2, y + key_height*0.8, key, 
                    ha='center', va='center', fontsize=30, color=main_color, 
                    fontweight='bold', zorder=2)
            
            # 添加声母信息
            if key in key_to_initial:
                initials = ','.join(key_to_initial[key])
                ax.text(x + key_width*0.3, y + key_height*0.5, 
                        initials, ha='center', va='center', 
                        fontsize=25, color=ini_color, zorder=2)
            
            # 添加韵母信息
            if key in key_to_final:
                finals = ','.join(key_to_final[key])
                ax.text(x + key_width*0.7, y + key_height*0.2, 
                        finals, ha='center', va='center', 
                        fontsize=25, color=fin_color, zorder=2)
    
    # 设置坐标轴范围 - 确保所有键盘都可见
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.5, total_rows * (key_height + key_spacing))
    
    # 隐藏坐标轴
    ax.axis('off')
    
    # 添加标题和图例
    plt.title('本双拼布局键位图', fontsize=16)
    
    # 创建图例
    ini_patch = patches.Patch(color=ini_color, label='声母')
    fin_patch = patches.Patch(color=fin_color, label='韵母')
    plt.legend(handles=[ini_patch, fin_patch], loc='upper center', 
               bbox_to_anchor=(0.5, -0.05), ncol=2)
    
    # 使用更通用的字体设置，解决缺少字形问题
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig('keyboard_layout.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"键盘键位图已保存为 'keyboard_layout.png'")

def main():
    """主函数"""
    # 默认文件路径
    prob_matrix_file = "pinyin_probability_matrix.txt"
    optimal_mapping_file = "optimal_mapping.json"
    
    # 检查文件是否存在
    if not os.path.exists(prob_matrix_file):
        print(f"错误: 找不到概率矩阵文件 '{prob_matrix_file}'")
        return
        
    if not os.path.exists(optimal_mapping_file):
        print(f"错误: 找不到优化双拼映射文件 '{optimal_mapping_file}'")
        return
    
    print("="*50)
    print("双拼布局测试台 - 开始生成对比图")
    print("="*50)
    
    # 绘制击键次数对比图
    print("\n1. 绘制击键次数对比图")
    draw_keystroke_comparison(prob_matrix_file, optimal_mapping_file)
    
    # 绘制击键流畅性对比图
    print("\n2. 绘制击键流畅性对比图")
    draw_fluency_comparison(prob_matrix_file, optimal_mapping_file)
    
    # 绘制手指负载均衡度对比图
    print("\n3. 绘制手指负载均衡度对比图")
    draw_load_balance_comparison(prob_matrix_file, optimal_mapping_file)
    
    # 绘制键盘键位图
    print("\n4. 绘制键盘键位图")
    draw_keyboard_layout(optimal_mapping_file)
    
    print("\n="*50)
    print("所有图表生成完成")
    print("="*50)

if __name__ == "__main__":
    main()
