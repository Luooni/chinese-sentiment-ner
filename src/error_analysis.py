# -*- coding: utf-8 -*-
"""
错误案例分析
=============
对比 Baseline 与 BERT 在测试集上的预测,分析:
1. BERT 判错但 Baseline 判对(模型之间的差异)
2. Baseline 判错但 BERT 判对(体现预训练模型优势)
3. 两者都判错(困难样本)

运行: python src/error_analysis.py
"""

import os
import json
import sys
import numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import jieba
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

LABELS = ["negative", "neutral", "positive"]
LABEL_ZH = {"negative": "负面", "neutral": "中性", "positive": "正面"}
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

STOPWORDS = set("""
的 了 和 是 在 我 你 他 她 它 们 这 那 有 也 都 就 不 而 及 与 着 或 又 很 太 更 最 还
一个 没有 我们 你们 他们 这个 那个 什么 怎么 为什么 因为 所以 但是 如果 虽然 然而 不过
还是 就是 可以 没有 非常 自己 已经 应该 需要 觉得 知道 时候 现在 这样 那样 之后 之前
一下 一点 一些 什么 东西 还是 不会 不能 不是 不要 这个 这些 那些 自己 就是 只是
""".split())


def load_test():
    with open(os.path.join(PROC_DIR, "sentiment_test.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    test = load_test()
    texts = [d["text"] for d in test]
    y_true = [d["label"] for d in test]
    print(f"测试集: {len(test)} 条")

    # --- Baseline 预测 ---
    print("加载 Baseline 模型...")
    clf = joblib.load(os.path.join(MODEL_DIR, "sentiment_baseline.pkl"))
    vec = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    X = vec.transform([" ".join([w for w in jieba.lcut(t) if w.strip() and w.strip() not in STOPWORDS])
                       for t in texts])
    y_base = clf.predict(X)

    # --- BERT 预测 ---
    print("加载 BERT 模型...")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "sentiment_bert"))
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(MODEL_DIR, "sentiment_bert"))
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    y_bert = []
    batch_size = 128
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                           padding=True, max_length=128)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        y_bert.extend([ID2LABEL[p] for p in preds])

    # --- 分类统计 ---
    both_wrong = []     # 两者都错
    bert_wrong_base_right = []  # BERT 错、Baseline 对
    base_wrong_bert_right = []  # Baseline 错、BERT 对

    for i, (t, gt) in enumerate(zip(texts, y_true)):
        b_ok = (y_base[i] == gt)
        b_bert_ok = (y_bert[i] == gt)
        if not b_bert_ok and not b_ok:
            both_wrong.append((t, gt, y_base[i], y_bert[i]))
        elif not b_bert_ok and b_ok:
            bert_wrong_base_right.append((t, gt, y_base[i], y_bert[i]))
        elif b_bert_ok and not b_ok:
            base_wrong_bert_right.append((t, gt, y_base[i], y_bert[i]))

    print("\n" + "=" * 60)
    print("错误案例统计")
    print("=" * 60)
    print(f"两者都错: {len(both_wrong)}")
    print(f"BERT 错、Baseline 对: {len(bert_wrong_base_right)}")
    print(f"Baseline 错、BERT 对: {len(base_wrong_bert_right)} (BERT 纠正的数量)")

    def show(title, cases, n=3):
        print(f"\n{'=' * 60}")
        print(f"{title} (共 {len(cases)} 条,展示 {min(n, len(cases))} 条)")
        print("=" * 60)
        for t, gt, b, bbert in cases[:n]:
            print(f"  文本: {t[:50]}{'...' if len(t) > 50 else ''}")
            print(f"    真实={LABEL_ZH[gt]}   Baseline={LABEL_ZH[b]}   BERT={LABEL_ZH[bbert]}")

    show("▍Baseline 判错、BERT 判对(BERT 优势)", base_wrong_bert_right)
    show("▍BERT 判错、Baseline 判对", bert_wrong_base_right)
    show("▍两者都判错(困难样本)", both_wrong)

    # 保存分析结果
    result = {
        "both_wrong": len(both_wrong),
        "bert_wrong_base_right": len(bert_wrong_base_right),
        "base_wrong_bert_right": len(base_wrong_bert_right),
    }
    with open(os.path.join(MODEL_DIR, "error_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n分析结果已保存: models/error_analysis.json")


if __name__ == "__main__":
    main()
