# 汉语双拼布局优化工具 (Fluent Type)

## 简介

本项目旨在通过优化汉语双拼输入法中声母和韵母到键盘按键的映射关系，来提升打字体验。当前许多双拼方案并未充分考虑指法效率，可能导致单指连续击键过多或手指负载不均的问题。

本工具的目标是在**不改变标准 QWERTY 物理键盘布局**的前提下，利用算法寻找更科学、更流畅的双拼映射方案，主要关注以下两点：

1.  **减少单指连续敲击不同按键**：降低打断输入节奏的情况。
2.  **均衡手指负载**：让各个手指的击键负担更加均匀。

## 核心功能

*   **语料处理与拼音生成**：从大规模中文语料库（如 `wiki_zh/`）中提取文本，并将其转换为拼音串。支持并行处理以加速。
*   **频率与概率矩阵生成**：基于生成的拼音串，统计“广义声母”与“广义韵母”的组合频率，并计算相应的概率矩阵。
*   **布局优化**：采用**模拟退火算法 (Simulated Annealing)**，以最小化自定义的损失函数（综合考虑单指连击和手指负载均衡）为目标，搜索最优的声母、韵母到键盘按键的映射方案。支持预设固定部分映射。
*   **可视化**：生成频率矩阵和概率矩阵的热力图，直观展示拼音组合的分布特性。
*   **结果评估**：输出最优映射方案，并分析该方案下的手指负载分布情况。

## 项目结构

```
.
├── fluent_type.py             # 主程序入口，整合所有流程
├── pinyin_generator.py        # 拼音串生成模块
├── matrix_generator.py        # 频率/概率矩阵生成与可视化模块
├── layout_optimizer.py        # 模拟退火布局优化器模块
├── parallel_processor.py      # 并行处理语料库模块
├── shuangpin_testbench.py     # (可能用于测试双拼方案，根据文件名推测)
├── wiki_zh/                   # 中文语料库文件夹 (示例)
├── pinyin_string.txt          # 生成的拼音串文件
├── pinyin_frequency_matrix.txt # 拼音频率矩阵数据
├── pinyin_probability_matrix.txt# 拼音概率矩阵数据
├── pinyin_frequency_matrix.png # 频率矩阵热力图
├── pinyin_probability_matrix.png# 概率矩阵热力图
├── pre_fixed_mapping.json     # 预设固定映射文件 (可选)
├── optimal_mapping.json       # 输出的最优映射文件
├── 产品规划.md                # 项目设计思路文档
├── readme.md                  # 本文档
└── ...                        # 其他辅助文件和图片
```

## 使用方法

### 环境要求

*   Python 3.x
*   必要的 Python 库：
    *   `numpy`
    *   `pypinyin`
    *   `opencc-python-reimplemented`
    *   `matplotlib`

    你可以使用 pip 安装它们：
    ```bash
    pip install numpy pypinyin opencc-python-reimplemented matplotlib
    ```

### 运行步骤

1.  **准备语料库**：将中文文本文件放入 `wiki_zh/` 文件夹，或者在运行时指定其他包含语料文件的文件夹路径或单个文件路径。语料库越大，统计结果越准确。
2.  **运行主程序**：
    ```bash
    python fluent_type.py [输入路径] [可选参数...]
    ```
    *   **`[输入路径]`** (可选): 指定语料库文件夹路径或单个文件路径。如果省略，默认为 `./wiki_zh`。
    *   **可选参数**:
        *   `[拼音文件]`: 输出拼音串的文件名 (默认: `pinyin_string.txt`)
        *   `[频率矩阵文件]`: 输出频率矩阵的文件名 (默认: `pinyin_frequency_matrix.txt`)
        *   `[频率图片]`: 输出频率热力图的文件名 (默认: `pinyin_frequency_matrix.png`)
        *   `[概率矩阵文件]`: 输出概率矩阵的文件名 (默认: `pinyin_probability_matrix.txt`)
        *   `[概率图片]`: 输出概率热力图的文件名 (默认: `pinyin_probability_matrix.png`)
        *   `[最大文件数]`: 处理的最大语料文件数 (默认: 200)
        *   `[每文件最大字符数]`: 每个文件处理的最大字符数 (默认: 50000)
        *   `[进程数]`: 并行处理使用的进程数 (默认: 自动使用所有 CPU 核心)

    **示例**:
    ```bash
    # 使用默认 wiki_zh 路径和默认参数运行
    python fluent_type.py

    # 指定语料文件夹 /data/corpus 和输出文件前缀 my_shuangpin
    python fluent_type.py /data/corpus my_shuangpin_pinyin.txt my_shuangpin_freq.txt my_shuangpin_freq.png my_shuangpin_prob.txt my_shuangpin_prob.png

    # 使用 wiki_zh，限制处理 100 个文件，每个文件最多 10000 字符，使用 4 个进程
    python fluent_type.py wiki_zh pinyin_string.txt pinyin_frequency_matrix.txt pinyin_frequency_matrix.png pinyin_probability_matrix.txt pinyin_probability_matrix.png 100 10000 4
    ```
    程序将依次执行拼音生成、矩阵计算、布局优化，并保存结果文件和图片。

3.  **(可选) 预设固定映射**: 如果你想固定某些声母或韵母的映射（例如，`zh`, `ch`, `sh` 必须映射到特定键），可以在 `pre_fixed_mapping.json` 文件中定义它们，格式如下：
    ```json
    {
        "initial_map": {
            "zh": "V",
            "ch": "I",
            "sh": "U"
        },
        "final_map": {
            "uang": "L",
            "iong": "S"
        }
    }
    ```
    优化器将保持这些映射不变，只优化剩余的映射。

### 独立运行优化器

如果你已经有了概率矩阵文件 (`pinyin_probability_matrix.txt`)，可以单独运行优化器：

```bash
python layout_optimizer.py --matrix <概率矩阵文件> --fixed <固定映射文件> --output <输出映射文件> --iter <迭代次数>
```

*   `--matrix` / `-m`: 指定概率矩阵文件路径 (默认: `pinyin_probability_matrix.txt`)。
*   `--fixed` / `-f`: 指定预设固定映射文件路径 (可选, 默认: `pre_fixed_mapping.json`)。
*   `--output` / `-o`: 指定最优映射输出文件路径 (默认: `optimal_mapping.json`)。
*   `--iter` / `-i`: 指定模拟退火的最大迭代次数 (默认: 20000)。

**示例**:
```bash
python layout_optimizer.py -m my_prob.txt -f my_fixed.json -o my_optimal.json -i 50000
```

## 算法说明

### 损失函数 (Loss Function)

损失函数用于评估一个双拼布局的好坏，由两部分组成：

1.  **Cost1 (单指连击代价)**: 衡量同一手指连续敲击不同按键的“不流畅度”。计算方式是遍历所有拼音组合 (声母+韵母)，如果其对应的两个按键由同一个手指负责敲击，则将该组合的出现概率累加到 Cost1 中（可加权）。
2.  **Cost2 (手指负载均衡度)**: 衡量主要击键手指（食指、中指、无名指、小指，共8根）的负载均衡程度。计算方式是统计每个手指负责的所有拼音组合的总概率，然后计算这8个手指负载值的**方差**。方差越小，表示负载越均衡。

**总损失 = Cost1 + w * Cost2** (w 是 Cost2 的权重，当前代码中 w=10)

目标是找到使总损失最小的声母/韵母映射方案。

### 模拟退火 (Simulated Annealing)

模拟退火是一种全局优化算法，用于在大搜索空间中寻找最优解。其过程如下：

1.  **初始化**: 生成一个随机的（或基于固定映射的）声母、韵母映射方案，计算初始损失。设置初始温度 T。
2.  **迭代**: 在每次迭代中：
    *   **扰动**: 随机选择一个声母或韵母（未被固定的），将其映射随机更改为另一个字母。
    *   **评估**: 计算新映射方案的损失值。
    *   **接受判断**:
        *   如果新损失更低，则接受新方案。
        *   如果新损失更高，则以一定概率接受新方案。这个概率随温度 T 的降低和损失差异的增大而减小 (Metropolis准则: `exp((旧损失 - 新损失) / T)`)。这使得算法有机会跳出局部最优解。
    *   **降温**: 缓慢降低温度 T (例如 `T *= 0.99999`)。
3.  **终止**: 当温度 T 低于某个阈值或达到最大迭代次数时，算法结束。记录下整个过程中找到的最佳映射方案和最低损失值。

## 作者

*   冯俊熙、GitHub Copilot
