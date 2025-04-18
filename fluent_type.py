# 汉语双拼布局优化工具
# 作者：GitHub Copilot

import os
import sys
import json

# 导入拼音生成器和矩阵生成器模块
import pinyin_generator as pg
import matrix_generator as mg

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
        print("正在处理语料库文件...")
        corpus_text = pg.load_corpus(corpus_file)
        
        if corpus_text:
            print(f"成功加载语料库，共 {len(corpus_text)} 个字符")
            
            # 生成拼音串
            print("正在生成拼音串...")
            pinyin_string = pg.get_pinyin_string(corpus_text)
            
            # 保存拼音串到文件
            if pg.save_pinyin_string(pinyin_string, pinyin_file):
                print("拼音串生成成功!")
                
                # 步骤2: 生成频率矩阵
                print("正在生成频率矩阵...")
                
                # 创建频率矩阵
                initials, finals, matrix = mg.create_frequency_matrix(pinyin_string)
                print(f"发现 {len(initials)} 个声母, {len(finals)} 个韵母")
                
                # 保存频率矩阵到文件
                if mg.save_matrix_csv(initials, finals, matrix, matrix_file):
                    print("频率矩阵已保存到文件")
                    
                    # 生成频率矩阵的可视化
                    print("正在生成频率矩阵可视化...")
                    mg.visualize_frequency_matrix(initials, finals, matrix, "语料库的声母-韵母频率矩阵", image_file)
                    print("可视化完成!")
                else:
                    print("保存频率矩阵失败")
            else:
                print("保存拼音串失败")
        else:
            print("语料库为空，请检查文件路径")
    else:
        # 如果没有提供文件路径，运行测试流程
        print("未提供语料库文件路径，运行测试...")
        test_text = "例如，这是你找到的字符串啊"
        
        # 生成拼音串
        print("正在生成拼音串...")
        pinyin_string = pg.get_pinyin_string(test_text)
        print("生成的拼音串结构:")
        print(pinyin_string)
        
        # 保存拼音串
        if pg.save_pinyin_string(pinyin_string):
            print("拼音串已保存到 pinyin_string.txt")
            
            # 生成频率矩阵
            print("正在生成频率矩阵...")
            initials, finals, matrix = mg.create_frequency_matrix(pinyin_string)
            print("声母列表:", initials)
            print("韵母列表:", finals)
            print("频率矩阵:")
            print(matrix)
            
            # 保存矩阵
            if mg.save_matrix_csv(initials, finals, matrix):
                print("频率矩阵已保存到 pinyin_frequency_matrix.txt")
                
                # 可视化
                print("正在生成频率矩阵可视化...")
                mg.visualize_frequency_matrix(initials, finals, matrix, "测试文本的声母-韵母频率矩阵")
                print("可视化完成!")
            else:
                print("保存频率矩阵失败")
        else:
            print("保存拼音串失败")

if __name__ == "__main__":
    main()