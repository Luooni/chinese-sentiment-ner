# 基于预训练模型的中文文本情感分析与信息抽取系统

一个完整的 NLP 系统,支持中文文本的情感三分类(正/负/中)与命名实体识别(人名/地名/机构/公司/产品/时间等),并对比传统方法与预训练模型的效果。

## 功能特性

- **文本预处理**:去噪、标准化、jieba 分词
- **情感分析**:三分类(负面/中性/正面)
  - Baseline:TF-IDF + 传统分类器(朴素贝叶斯/逻辑回归/SVM)
  - 进阶:`bert-base-chinese` 微调
- **信息抽取**:10 类命名实体识别,`bert-base-chinese` 序列标注微调
- **可视化**:词云、情感分布、实体类型统计(Streamlit 交互界面)
- **对比实验**:传统方法 vs BERT,含准确率/F1/混淆矩阵/错误案例分析

## 目录结构

```
.
├── data/
│   ├── raw/              原始数据(CLUENER2020、amazon_reviews_multi)
│   ├── processed/        预处理后数据(统一 JSON 格式)
│   └── external/         外部资源(停用词表)
├── src/
│   ├── preprocess.py      数据预处理
│   ├── sentiment/
│   │   ├── baseline.py    TF-IDF + 传统分类器
│   │   └── bert_train.py  BERT 微调
│   ├── ner/
│   │   └── bert_ner.py    BERT 序列标注微调
│   ├── evaluate.py        对比实验与图表
│   └── error_analysis.py  错误案例分析
├── models/                训练好的模型
├── app.py                 Streamlit 交互界面
├── report/                实验报告与图表
└── requirements.txt       依赖清单
```

## 安装与运行

```bash
pip install -r requirements.txt

# 1. 数据预处理
python src/preprocess.py

# 2. 训练 Baseline
python src/sentiment/baseline.py

# 3. 训练 BERT 情感分类
python src/sentiment/bert_train.py

# 4. 训练 BERT 命名实体识别
python src/ner/bert_ner.py

# 5. 对比实验与错误分析
python src/evaluate.py
python src/error_analysis.py

# 6. 启动交互界面
streamlit run app.py
```

## 数据集

| 任务 | 数据集 | 规模 | 说明 |
|------|--------|------|------|
| 情感分析 | amazon_reviews_multi(中文) | 训练 20 万条 | 1-5 星映射为三分类(1-2 负/3 中/4-5 正) |
| 实体识别 | CLUENER2020 | 训练 10748 条 | 10 类细粒度中文实体 |

## 技术路线

| 模块 | 基础方案 | 进阶方案 |
|------|----------|----------|
| 分词 | jieba | jieba |
| 情感分析 | TF-IDF + 逻辑回归 | BERT 微调 |
| 实体识别 | — | BERT + 序列标注(BIO) |
| 可视化 | matplotlib + WordCloud | Streamlit |
| 评估 | 准确率、F1 | 混淆矩阵、错误分析 |

> 注:HuggingFace 模型通过国内镜像 `hf-mirror.com` 下载,脚本内已设置 `HF_ENDPOINT` 环境变量。
