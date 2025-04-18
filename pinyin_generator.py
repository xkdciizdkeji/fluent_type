# 拼音生成器
# 作者：GitHub Copilot

import re
import string
import json
from pypinyin import pinyin, Style, lazy_pinyin

def is_punctuation(char):
    """判断一个字符是否为标点符号"""
    return char in string.punctuation or char in "，。！？；：、""''（）【】《》…—"

def split_pinyin(py):
    """
    将一个拼音拆分为声母和韵母
    对于单韵母字，如"a"，返回["a", "a"]
    对于有声母有韵母的字，如"ping"，返回["p", "ing"]
    """
    # 声母列表
    initials = ['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 
                'zh', 'ch', 'sh', 'r', 'z', 'c', 's', 'y', 'w']
    
    # 对于空字符串，返回空列表
    if not py:
        return ["", ""]
    
    # 检查是否以声母开头
    for i in initials:
        if py.startswith(i):
            # 有声母，返回[声母, 韵母]
            return [i, py[len(i):]]
    
    # 无声母（如"a", "o", "e"等），我们假装它有一个与自己相同的声母
    return [py, py]

def get_pinyin_string(text):
    """
    将中文文本转换为拼音串
    返回格式为：[[["声母", "韵母"], ["声母", "韵母"]], ...]
    其中第一级是以标点符号为分隔的句子，第二级是句子中的字，第三级是每个字的声母和韵母
    """
    # 使用标点符号将文本分割成句子
    # 先将所有标点符号替换为特殊标记
    processed_text = text
    for char in text:
        if is_punctuation(char):
            processed_text = processed_text.replace(char, "###PUNCT###")
    
    # 按特殊标记分割文本
    sentences = processed_text.split("###PUNCT###")
    # 过滤掉空字符串
    sentences = [s.strip() for s in sentences if s.strip()]
    
    result = []
    for sentence in sentences:
        # 获取每个汉字的拼音（小写，不带声调）
        pys = lazy_pinyin(sentence, style=Style.NORMAL)
        
        # 将每个拼音拆分为声母和韵母
        pinyin_pairs = [split_pinyin(py) for py in pys]
        
        # 只添加非空的拼音对
        sentence_result = [pair for pair in pinyin_pairs if pair[0] or pair[1]]
        if sentence_result:
            result.append(sentence_result)
    
    return result

def load_corpus(file_path):
    """
    加载语料库文件
    
    Args:
        file_path: 语料库文件路径
    
    Returns:
        文本内容
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"加载语料库失败: {e}")
        return ""

def save_pinyin_string(pinyin_string, output_file="pinyin_string.txt"):
    """
    保存拼音串到文件
    
    Args:
        pinyin_string: 拼音串
        output_file: 输出文件路径
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pinyin_string, f, ensure_ascii=False, indent=2)
        print(f"拼音串已保存到 {output_file}")
        return True
    except Exception as e:
        print(f"保存拼音串失败: {e}")
        return False

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1:
        # 如果提供了文件路径，加载文件
        file_path = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "pinyin_string.txt"
        
        print(f"正在处理语料库文件: {file_path}")
        corpus_text = load_corpus(file_path)
        
        if corpus_text:
            print(f"成功加载语料库，共 {len(corpus_text)} 个字符")
            pinyin_string = get_pinyin_string(corpus_text)
            save_pinyin_string(pinyin_string, output_file)
        else:
            print("语料库为空，请检查文件路径")
    else:
        # 如果没有提供文件路径，运行测试
        print("未提供语料库文件路径，运行测试...")
        test_text = "例如，这是你找到的字符串啊"
        pinyin_string = get_pinyin_string(test_text)
        print(pinyin_string)
        save_pinyin_string(pinyin_string)

if __name__ == "__main__":
    main()