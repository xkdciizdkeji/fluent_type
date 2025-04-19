
import re
import string
import json
import os
from pypinyin import pinyin, Style, lazy_pinyin

# 预编译正则表达式
CHINESE_REGEX = re.compile('[\u4e00-\u9fff]')
PUNCT_REGEX = re.compile(f'[{re.escape(string.punctuation)}，。！？；：、""''（）【】《》…—]')
PUNCT_SPLIT_REGEX = re.compile(f'[{re.escape(string.punctuation)}，。！？；：、""''（）【】《》…—]+')
NON_CHINESE_REGEX = re.compile('[^\u4e00-\u9fff]')

# 预定义声母列表，并构建查找字典以加速查询
INITIALS = ['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 
            'zh', 'ch', 'sh', 'r', 'z', 'c', 's', 'y', 'w']
INITIALS_DICT = {i: True for i in INITIALS}
# 两字母和三字母声母的列表，用于快速拆分
TWO_LETTER_INITIALS = ['zh', 'ch', 'sh']

# 预编译汉字到声母韵母的缓存表，用于常用字
PINYIN_CACHE = {}

def is_punctuation(char):
    """判断一个字符是否为标点符号（使用正则表达式更快）"""
    return bool(PUNCT_REGEX.match(char))

def split_pinyin(py):
    """
    将一个拼音拆分为声母和韵母（优化版）
    
    处理规则：
    1. 对于带声母的拼音，如"ping"，返回["p", "ing"]
    2. 对于单韵母字，如"a"，返回["a", "a"]
    3. 对于双字母韵母字，如"en"，返回["e", "n"]
    4. 对于三字母或以上的韵母字，如"ang"，返回["a", "ang"]
    
    Args:
        py: 拼音字符串
    
    Returns:
        [声母, 韵母]列表
    """
    # 对于空字符串，返回空列表
    if not py:
        return ["", ""]
    
    # 首先检查两字母声母
    if len(py) >= 2:
        possible_initial = py[:2]
        if possible_initial in TWO_LETTER_INITIALS:
            return [possible_initial, py[2:]]
    
    # 然后检查单字母声母（使用字典查找更快）
    first_char = py[0]
    if first_char in INITIALS_DICT:
        return [first_char, py[1:]]
    
    # 无声母情况（纯韵母）
    # 1. 单字母韵母，如"a"、"o"、"e"等
    if len(py) == 1:
        return [py, py]
    
    # 2. 双字母韵母，如"ai"、"ei"、"ou"、"en"等
    elif len(py) == 2:
        return [py[0], py[1]]
    
    # 3. 三字母或以上的韵母，如"ang"、"eng"、"iang"等
    else:
        return [py[0], py[0:]]

def get_pinyin_string(text):
    """
    将中文文本转换为拼音串（优化版）
    返回格式为：[[["声母", "韵母"], ["声母", "韵母"]], ...]
    其中第一级是以标点符号为分隔的句子，第二级是句子中的字，第三级是每个字的声母和韵母
    """
    # 如果输入为空，返回空列表
    if not text:
        return []
    
    # 使用正则表达式检查文本是否包含中文字符
    if not CHINESE_REGEX.search(text):
        # 非中文文本处理，作为一个句子
        words = text.split()
        if not words:
            words = [text]
        
        sentence_result = []
        for word in words:
            # 只处理非空字符
            if word.strip():  
                for char in word.strip():
                    if char.strip():  # 忽略空白字符
                        pair = split_pinyin(char.lower())
                        if pair[0] or pair[1]:
                            # 检测异常拼音
                            if "兙" in pair[0] or "兙" in pair[1]:
                                print(f"警告: 字符 '{char}' 产生了异常拼音: {pair}")
                            sentence_result.append(pair)
        
        return [sentence_result] if sentence_result else []
    
    # 处理包含中文的文本
    # 使用正则表达式直接切分句子（比逐字替换更快）
    sentences = PUNCT_SPLIT_REGEX.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    result = []
    
    for sentence in sentences:
        # 移除句子中的非中文字符以加速处理，只保留可能转换为拼音的中文字符
        clean_sentence = NON_CHINESE_REGEX.sub('', sentence)
        
        if not clean_sentence:
            continue
        
        # 使用缓存加速重复字符的处理
        sentence_pairs = []
        
        # 获取拼音（使用lazy_pinyin为最快的方法）
        pys = lazy_pinyin(clean_sentence, style=Style.NORMAL)
        
        # 批量处理拼音对
        for i, py in enumerate(pys):
            # 检查缓存
            char = clean_sentence[i]
            
            # 直接检测原始拼音是否含有异常字符
            if "兙" in py:
                print(f"警告: 汉字 '{char}' (位置 {i}) 的原始拼音 '{py}' 包含异常字符")
            
            # 直接检测原始拼音是否含有空声母或韵母
            if py == "":
                print(f"警告: 汉字 '{char}' (位置 {i}) 的原始拼音 '{py}' 是空字符串")
                
            if char in PINYIN_CACHE:
                pair = PINYIN_CACHE[char]
            else:
                pair = split_pinyin(py)
                # 缓存结果以加速将来的查询
                PINYIN_CACHE[char] = pair
            
            # 检测异常拼音
            if "兙" in pair[0] or "兙" in pair[1]:
                print(f"警告: 汉字 '{char}' 的拼音对 {pair} 包含异常字符 '兙'")
                print(f"      原始拼音: '{py}'，Unicode值: {[ord(c) for c in py]}")
                print(f"      字符Unicode: {ord(char)}, 十六进制: {hex(ord(char))}")
                # 跳过这个异常拼音对
                continue

            # 检测空声母或韵母
            if pair[0] == "" or pair[1] == "":
                print(f"警告: 汉字 '{char}' 的拼音对 {pair} 是空字符串")
                continue
            
            # 只添加非空对
            if pair[0] or pair[1]:
                sentence_pairs.append(pair)
        
        if sentence_pairs:
            result.append(sentence_pairs)
    
    return result

def init_pinyin_cache(common_chars=None):
    """
    预先初始化常用汉字的拼音缓存
    
    Args:
        common_chars: 常用汉字字符串，如果不提供，将使用内置的3000个常用汉字
    """
    global PINYIN_CACHE
    
    if common_chars is None:
        # 内置的常用汉字（前几百个常用字）
        common_chars = "的一是不了在人有我他这个们中来上大为和国地到以说时要就出会可也你对生能而子那得于着下自之年过发后作里用道行所然家种事成方多经么去法学如都同现当没动面起看定天分还进好小部其些主样理心她本前开但因只从想实日军者意无力它与长把机十民第公此已工使情明性知全三又关点正业外将两高间由问很最重并物手应战向头文体政美相见被利什二等产或新己制身果加西斯民领己世候市动号妈各好称孩纸数给景真因战路总被教认况使十气"
    
    # 批量获取拼音，然后进行拆分
    pys = lazy_pinyin(common_chars, style=Style.NORMAL)
    
    for i, char in enumerate(common_chars):
        if char not in PINYIN_CACHE:
            PINYIN_CACHE[char] = split_pinyin(pys[i])

# 在模块加载时初始化常用汉字的拼音缓存
init_pinyin_cache()

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

def load_corpus_folder(folder_path, max_files=None, max_chars=None):
    """
    递归加载文件夹中的所有文本文件
    
    Args:
        folder_path: 文件夹路径
        max_files: 最大加载的文件数量，默认为None表示不限制
        max_chars: 最大加载的字符数量，默认为None表示不限制
    
    Returns:
        文本内容
    """
    if not os.path.isdir(folder_path):
        print(f"错误: {folder_path} 不是一个有效的文件夹")
        return ""
    
    all_text = ""
    files_loaded = 0
    total_chars = 0
    
    # 遍历所有子文件夹和文件
    for root, _, files in os.walk(folder_path):
        for file in files:
            # 跳过隐藏文件和临时文件
            if file.startswith('.') or file.endswith('~'):
                continue
            
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    file_content = f.read()
                    
                    # 限制字符数量
                    if max_chars and total_chars + len(file_content) > max_chars:
                        remaining = max_chars - total_chars
                        if remaining > 0:
                            all_text += file_content[:remaining]
                            total_chars += remaining
                        print(f"已达到最大字符数限制 ({max_chars} 字符)")
                        return all_text
                    
                    all_text += file_content
                    total_chars += len(file_content)
                    files_loaded += 1
                    
                    # 定期打印进度
                    if files_loaded % 10 == 0:
                        print(f"已加载 {files_loaded} 个文件, 共 {total_chars} 个字符")
                    
                    # 限制文件数量
                    if max_files and files_loaded >= max_files:
                        print(f"已达到最大文件数限制 ({max_files} 个文件)")
                        return all_text
                    
            except Exception as e:
                print(f"无法加载文件 {file_path}: {e}")
    
    print(f"加载完成，共处理了 {files_loaded} 个文件, {total_chars} 个字符")
    return all_text

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

# 当作为独立程序运行时的入口点
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "pinyin_string.txt"
        max_files = int(sys.argv[3]) if len(sys.argv) > 3 else None
        max_chars = int(sys.argv[4]) if len(sys.argv) > 4 else None
        
        print(f"正在处理路径: {path}")
        
        # 检查是文件还是文件夹
        if os.path.isdir(path):
            print(f"检测到文件夹，将递归加载所有文件...")
            corpus_text = load_corpus_folder(path, max_files, max_chars)
        else:
            print(f"检测到单个文件，直接加载...")
            corpus_text = load_corpus(path)
        
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
        print("生成的拼音串结构:")
        print(pinyin_string)
        save_pinyin_string(pinyin_string)