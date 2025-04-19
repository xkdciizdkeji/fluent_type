
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

def create_probability_matrix(initials, finals, freq_matrix):
    """
    将频率矩阵转换为概率矩阵
    
    Args:
        initials: 声母列表
        finals: 韵母列表
        freq_matrix: 声母-韵母频率矩阵
    
    Returns:
        prob_matrix: 概率矩阵
        total_count: 总次数
    """
    # 计算总次数
    total_count = np.sum(freq_matrix)
    
    # 如果总次数为0，返回全0矩阵
    if total_count == 0:
        return np.zeros_like(freq_matrix), 0
    
    # 创建概率矩阵
    prob_matrix = freq_matrix / total_count
    
    return prob_matrix, total_count

def visualize_frequency_matrix(initials, finals, matrix, title="声母-韵母频率矩阵", output_file="pinyin_frequency_matrix.png", show=True):
    """
    将频率矩阵可视化为热力图
    
    Args:
        initials: 声母列表
        finals: 韵母列表
        matrix: 声母-韵母频率矩阵
        title: 图表标题
        output_file: 输出文件路径
        show: 是否显示图表
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
    
    if show:
        plt.show()
    else:
        plt.close()

def visualize_probability_matrix(initials, finals, matrix, title="声母-韵母概率矩阵", output_file="pinyin_probability_matrix.png", show=True):
    """
    将概率矩阵可视化为热力图
    
    Args:
        initials: 声母列表
        finals: 韵母列表
        matrix: 声母-韵母概率矩阵
        title: 图表标题
        output_file: 输出文件路径
        show: 是否显示图表
    """
    plt.figure(figsize=(16, 12))
    
    # 创建热力图，使用对数刻度显示概率（加小数避免对0取对数）
    log_matrix = np.log1p(matrix * 1000)  # 乘以1000使小概率更明显
    sns.heatmap(log_matrix, annot=False, fmt=".2f", xticklabels=finals, yticklabels=initials, cmap="YlGnBu")
    
    plt.title(title, fontsize=16)
    plt.xlabel("韵母", fontsize=14)
    plt.ylabel("声母", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_file, dpi=300)
    print(f"概率矩阵热力图已保存到 {output_file}")
    
    if show:
        plt.show()
    else:
        plt.close()

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

def save_probability_matrix_csv(initials, finals, matrix, output_file="pinyin_probability_matrix.txt"):
    """
    将概率矩阵保存为CSV/TXT格式
    
    Args:
        initials: 声母列表
        finals: 韵母列表
        matrix: 声母-韵母概率矩阵
        output_file: 输出文件路径
    """
    try:
        # 保存为带标题的CSV格式
        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入标题行（韵母）
            f.write("\t" + "\t".join(finals) + "\n")
            
            # 写入每一行（声母及对应的概率）
            for i, initial in enumerate(initials):
                row = [initial] + [f"{matrix[i, j]:.6f}" for j in range(len(finals))]
                f.write("\t".join(row) + "\n")
                
        print(f"概率矩阵已保存到 {output_file}")
        return True
    except Exception as e:
        print(f"保存概率矩阵失败: {e}")
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

# 当作为独立程序运行时的入口点
if __name__ == "__main__":
    import sys
    
    # 默认输入输出文件
    input_file = "pinyin_string.txt"
    freq_matrix_file = "pinyin_frequency_matrix.txt"
    prob_matrix_file = "pinyin_probability_matrix.txt"
    freq_image_file = "pinyin_frequency_matrix.png"
    prob_image_file = "pinyin_probability_matrix.png"
    
    # 处理命令行参数
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        freq_matrix_file = sys.argv[2]
    if len(sys.argv) > 3:
        freq_image_file = sys.argv[3]
    if len(sys.argv) > 4:
        prob_matrix_file = sys.argv[4]
    if len(sys.argv) > 5:
        prob_image_file = sys.argv[5]
    
    print(f"正在从 {input_file} 加载拼音串...")
    pinyin_strings = load_pinyin_string(input_file)
    
    if pinyin_strings:
        print("成功加载拼音串")
        
        # 生成频率矩阵
        initials, finals, freq_matrix = create_frequency_matrix(pinyin_strings)
        
        # 生成概率矩阵
        prob_matrix, total_count = create_probability_matrix(initials, finals, freq_matrix)
        
        print(f"发现 {len(initials)} 个声母, {len(finals)} 个韵母, 总计 {int(total_count)} 个拼音")
        
        print("保存频率矩阵数据...")
        save_matrix_csv(initials, finals, freq_matrix, freq_matrix_file)
        
        print("保存概率矩阵数据...")
        save_probability_matrix_csv(initials, finals, prob_matrix, prob_matrix_file)
        
        print("生成频率矩阵可视化...")
        visualize_frequency_matrix(initials, finals, freq_matrix, "拼音声母韵母频率矩阵", freq_image_file)
        
        print("生成概率矩阵可视化...")
        visualize_probability_matrix(initials, finals, prob_matrix, "拼音声母韵母概率矩阵", prob_image_file)
    else:
        print("拼音串为空，请检查文件路径")