# 数据清洗思路与难点日志 (Data Cleaning Notes)

这份文档专门为你准备，用来应对老师/助教复盘时关于**“数据清洗和预处理（Data Cleaning & Preprocessing）”**的提问。

---

## 中文解析 (Chinese with English Terminology)

### 核心痛点与问题现象
当我们从 Hugging Face 导入大规模游戏数据集时，**数据集里关于 "Genres", "Tags", "Categories" 的元数据（Metadata）并不是标准格式**。
原始数据往往是以 `list` （数组）形态压缩成了没有任何逻辑的纯字符串，例如：`"['Action' 'Adventure']"` 或者 `"['RPG', 'Indie']"`。 

如果跳过数据清洗直接喂给 Streamlit 前端或者交给机器学习模型：
1. **维度爆炸 (Dimension Explosion)**：前端 UI 会把 `['Action', 'RPG']` 整个长句误认为是一个单独且唯一的类型，导致侧边栏的 Filter 多出成千上万个毫无意义的重叠勾选项。
2. **影响聚类与词向量计算**：标点符号（方括号、单引号）会造成极大噪音。

### 我们采取的 2 步清洗策略

#### 第一步：正则表达式剥离 (Regex Unpacking) —— 发生在于 `data.py`
在数据使用 pandas DataFrame 生成的那一刻，我们应用了自定义的 `clean_hf_list()` 函数：
- **原理**：使用正则表达式 `re.findall(r"['\"]([^'\"]+)['\"], val_str)` 精准穿透方括号，仅把处于单引号或双引号内部的“纯单词”提取出来。
- **降级机制（Fallback）**：处理那些没有被单引号包裹的坏数据，暴力替换掉所有的 `[` 和 `]`。
- **输出结果**：所有的脏字符串阵列被格式化为极其干练的**逗号分隔自然文本**（例如：`Action, Adventure, RPG`）。

#### 第二步：利用 Hash 集合自动去重并合并概念 (Set Deduplication) —— 发生在于 `app.py`
当前端需要渲染到底有哪些选项存在时：
- **原理**：我们遍历清洗后的逗号字符串，将它们按逗号拆分出单个词：`['Action', 'RPG']`。
- **去重算法**：声明了一个通过哈希表（Hash Table）底层实现的 Python 集合——`set()`。在计算机科学中，`set` 的数学定义严格保障了**容器内绝对不允许同名元素存在**。
- **效果**：无论 5000 款游戏里的文本被拆解生成了多少万个标签词，每当遇到相同的词汇（如千万次出现的 `Action`）试图装入 `set`，它们会被立刻合并（$O(1)$ 时间复杂度）。这使得侧边栏奇迹般地仅仅只剩下了十几个最基础的游戏母类别，极大地优化了用户体验和交互性能。

---

## English Version

### Core Pain Points & Issues
When importing large-scale gaming datasets from Hugging Face, the metadata referencing "Genres", "Tags", and "Categories" is fundamentally unformatted.
The raw data is stored as list structures compressed directly into flat, illogical string entities, such as `"['Action' 'Adventure']"` or `"['RPG', 'Indie']"`.

If we skip data cleaning and feed this raw data directly into the Streamlit frontend or machine learning models:
1. **Dimension Explosion**: The UI filter will mistakenly treat long strings like `['Action', 'RPG']` as an entirely unique and single category, causing the sidebar filter to bloat with thousands of meaningless, overlapping checkboxes.
2. **Clustering & Embedding Interference**: Punctuation marks (brackets, quotes) introduce extreme noise to tokenizers and metric calculations.

### Our 2-Step Cleaning Strategy

#### Step 1: Regex Unpacking (Executed in `data.py`)
At the exact moment the data is materialized into a pandas DataFrame, we deployed a custom `clean_hf_list()` function:
- **Theory**: We utilized a Regular Expression (`re.findall(r"['\"]([^'\"]+)['\"], val_str)`) to pierce the brackets precisely, extracting only the "pure words" wrapped inside the single or double quotes.
- **Fallback Mechanism**: To handle malformed data completely missing proper quotation marks, we forcefully strip all `[` and `]` characters.
- **Output Result**: All dirty array strings are heavily formatted into pristine **comma-separated natural text** (e.g., `Action, Adventure, RPG`).

#### Step 2: Set Deduplication via Hash Tables (Executed in `app.py`)
When the frontend needs to render the exact available filtering options:
- **Theory**: We iterate over the cleaned comma-separated string, splitting them into individual words: `['Action', 'RPG']`.
- **Deduplication Algorithm**: We declared a Python `set()`, which is structurally engineered upon Hash Tables. In computer science, a `set` mathematically guarantees that **no identically named elements are permitted to exist concurrently within the container**.
- **Impact**: Regardless of whether extracting words from 5,000 games generates hundreds of thousands of individual tags—every time an identical word (e.g., millions of occurrences of `Action`) attempts to insert itself into the `set`, they are instantly merged ($O(1)$ Time Complexity). This miraculously slims down the sidebar filter to only display a dozen fundamental gaming categories, massively optimizing the User Experience and interactive latency.
