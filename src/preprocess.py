# -*- coding: utf-8 -*-
"""
数据预处理脚本
================
功能:
1. 情感分析数据:amazon_reviews_multi 中文评论 -> 清洗 + 三分类(正/负/中) + 统一 JSON 格式
2. 命名实体识别:CLUENER2020 -> 转换为 BIO 标注格式

运行: python src/preprocess.py
"""

import os
import re
import json
import gzip
import sys
from collections import Counter

# Windows 控制台输出中文
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 路径配置(相对项目根目录)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

# 每类情感样本的最大训练数量(平衡采样,保证正/负/中均衡)
MAX_PER_CLASS = 20000


def clean_text(text: str) -> str:
    """清洗文本:去 HTML、URL、特殊符号、多余空白"""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)                       # 去除 HTML 标签
    text = re.sub(r"http\S+|www\.\S+", "", text)              # 去除 URL
    text = re.sub(r"[^一-龥a-zA-Z0-9\s。，！？、；：""''（）…—··]", "", text)  # 保留中文/字母/数字/常用标点
    text = re.sub(r"\s+", " ", text).strip()                  # 合并空白
    return text


def stars_to_label(stars: int) -> str:
    """星级 -> 三分类标签:1-2 负 / 3 中 / 4-5 正"""
    if stars <= 2:
        return "negative"
    if stars == 3:
        return "neutral"
    return "positive"


def preprocess_sentiment() -> None:
    """处理情感分析数据,输出 data/processed/sentiment_{split}.json"""
    print("=" * 50)
    print("[情感分析] 数据预处理")
    for split in ["train", "validation", "test"]:
        src = os.path.join(RAW_DIR, f"amazon_zh_{split}.jsonl.gz")
        out_path = os.path.join(PROC_DIR, f"sentiment_{split}.json")

        counter = Counter()   # 各类别计数(用于平衡采样)
        data = []
        seen = set()          # 去重集合

        with gzip.open(src, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                title = clean_text(d.get("review_title", ""))
                body = clean_text(d.get("review_body", ""))
                text = (title + " " + body).strip() if title else body

                # 过滤过短文本
                if len(text) < 5:
                    continue
                # 去重
                if text in seen:
                    continue
                seen.add(text)

                label = stars_to_label(d["stars"])

                # 训练集做平衡采样(每类最多 MAX_PER_CLASS)
                if split == "train" and counter[label] >= MAX_PER_CLASS:
                    continue
                counter[label] += 1

                data.append({"text": text, "label": label})

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)

        print(f"  {split:10s} -> {len(data):>6} 条  分布: {dict(counter)}")


# CLUENER2020 实体类型 -> 题目要求对应关系(共 10 类)
NER_TAGS = [
    "name",         # 人名
    "address",      # 地名
    "organization", # 机构名
    "company",      # 公司
    "game",         # 游戏
    "movie",        # 电影
    "book",         # 书籍
    "government",   # 政府
    "position",     # 职位
    "scene",        # 场景
]


def convert_to_bio(text: str, label_dict: dict):
    """
    将 CLUENER2020 的实体标注转为 BIO 序列。
    label_dict: {"实体类型": {"实体文本": [[start, end], ...]}}
    返回: (tokens, bio_tags)  其中 tokens 为字符列表
    """
    tags = ["O"] * len(text)
    for etype, entities in label_dict.items():
        for spans in entities.values():
            for start, end in spans:
                # 闭区间 [start, end]
                for i in range(start, end + 1):
                    if i >= len(text):
                        continue
                    if i == start:
                        tags[i] = f"B-{etype}"
                    else:
                        # 若已被标记则跳过,避免嵌套覆盖
                        if tags[i] == "O":
                            tags[i] = f"I-{etype}"
    return list(text), tags


def preprocess_ner() -> None:
    """处理 CLUENER2020,输出 BIO 标注数据 data/processed/ner_{split}.json"""
    print("=" * 50)
    print("[实体识别] CLUENER2020 -> BIO 标注")
    for split in ["train", "dev"]:
        src = os.path.join(RAW_DIR, "cluener2020", f"{split}.json")
        out_path = os.path.join(PROC_DIR, f"ner_{split}.json")

        data = []
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                chars, tags = convert_to_bio(d["text"], d.get("label", {}))
                data.append({"text": d["text"], "chars": chars, "tags": tags})

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)

        # 统计实体数量
        ent_counter = Counter()
        for sample in data:
            for t in sample["tags"]:
                if t.startswith("B-"):
                    ent_counter[t[2:]] += 1
        print(f"  {split:5s} -> {len(data):>6} 条  实体分布: {dict(ent_counter)}")

    # 处理无标签的 test 集(仅保留文本,供预测演示)
    src = os.path.join(RAW_DIR, "cluener2020", "test.json")
    out_path = os.path.join(PROC_DIR, "ner_test.json")
    data = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            data.append({"text": d["text"]})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"  test  -> {len(data):>6} 条 (无标签,用于预测演示)")


if __name__ == "__main__":
    preprocess_sentiment()
    preprocess_ner()
    print("\n预处理完成!输出目录:", PROC_DIR)
