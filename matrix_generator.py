# 拼音概率矩阵生成器
# 作者：GitHub Copilot

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

def create_frequency_matrix(pinyin_strings):
    """
    根据拼音串生成声母-韵母频率矩阵
    
    Args:
        pinyin_strings: 格式为[[["声母", "韵母"], ["声母", "韵母"]], ...]的拼音串
    
    Returns:
        initials: 声母列表
        finals: 韵母列表
        matrix: 声母-韵母频率矩阵
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
    
    # 将集合转换为排序后的列表
    initials = sorted(list(initials_set))
    finals = sorted(list(finals_set))
    
    # 创建频率矩阵
    matrix = np.zeros((len(initials), len(finals)))
    
    # 填充矩阵
    for i, initial in enumerate(initials):
        for j, final in enumerate(finals):
            matrix[i, j] = initial_final_counts.get((initial, final), 0)
    
    return initials, finals, matrix

def visualize_frequency_matrix(initials, finals, matrix, title="声母-韵母频率矩阵", output_file="pinyin_frequency_matrix.png"):
    """
    将频率矩阵可视化为热力图
    
    Args:
        initials: 声母列表
        finals: 韵母列表
        matrix: 声母-韵母频率矩阵
        title: 图表标题
        output_file: 输出文件路径
    """
    plt.figure(figsize=(16, 12))
    
    # 创建热力图，使用对数刻度显示频率
    log_matrix = np.log1p(matrix)  # log1p避免对0取对数
    sns.heatmap(log_matrix, annot=False, fmt=".2f", xticklabels=finals, yticklabels=initials, cmap="YlGnBu")
    
    plt.title(title, fontsize=16)
    plt.xlabel("韵母", fontsize=14)
    plt.ylabel("声母", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_file, dpi=300)
    print(f"频率矩阵热力图已保存到 {output_file}")
    plt.show()

def save_matrix_data(initials, finals, matrix, output_file="matrix_data.json"):
    """
    将矩阵数据保存到文件
    
    Args:
        initials: 声母列表
        finals: 韵母列表
        matrix: 声母-韵母频率矩阵
        output_file: 输出文件路径
    """
    data = {
        "initials": initials,
        "finals": finals,
        "matrix": matrix.tolist()
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"矩阵数据已保存到 {output_file}")
        return True
    except Exception as e:
        print(f"保存矩阵数据失败: {e}")
        return False

def save_matrix_csv(initials, finals, matrix, output_file="pinyin_frequency_matrix.txt"):
    """
    将矩阵保存为CSV/TXT格式
    
    Args:
        initials: 声母列表
        finals: 韵母列表
        matrix: 声母-韵母频率矩阵
        output_file: 输出文件路径
    """
    try:
        # 保存为带标题的CSV格式
        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入标题行（韵母）
            f.write("\t" + "\t".join(finals) + "\n")
            
            # 写入每一行（声母及对应的频率）
            for i, initial in enumerate(initials):
                row = [initial] + [str(int(matrix[i, j])) for j in range(len(finals))]
                f.write("\t".join(row) + "\n")
                
        print(f"矩阵已保存到 {output_file}")
        return True
    except Exception as e:
        print(f"保存矩阵失败: {e}")
        return False

def load_pinyin_string(file_path="pinyin_string.txt"):
    """
    从文件加载拼音串
    
    Args:
        file_path: 文件路径
    
    Returns:
        拼音串
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载拼音串失败: {e}")
        return []

def main():
    """主函数"""
    import sys
    
    # 默认输入输出文件
    input_file = "pinyin_string.txt"
    matrix_file = "pinyin_frequency_matrix.txt"
    image_file = "pinyin_frequency_matrix.png"
    
    # 处理命令行参数
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        matrix_file = sys.argv[2]
    if len(sys.argv) > 3:
        image_file = sys.argv[3]
    
    print(f"正在从 {input_file} 加载拼音串...")
    pinyin_strings = load_pinyin_string(input_file)
    
    if pinyin_strings:
        print("成功加载拼音串")
        initials, finals, matrix = create_frequency_matrix(pinyin_strings)
        
        print(f"发现 {len(initials)} 个声母, {len(finals)} 个韵母")
        print("保存频率矩阵数据...")
        save_matrix_csv(initials, finals, matrix, matrix_file)
        
        print("生成频率矩阵可视化...")
        visualize_frequency_matrix(initials, finals, matrix, "拼音声母韵母频率矩阵", image_file)
    else:
        print("拼音串为空，请检查文件路径")

if __name__ == "__main__":
    main()