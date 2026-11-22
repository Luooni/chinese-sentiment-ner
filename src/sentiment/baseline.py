# -*- coding: utf-8 -*-
"""
情感分析 Baseline:分词 + TF-IDF + 传统机器学习分类器
====================================================
对比三种传统分类器:朴素贝叶斯、逻辑回归、线性 SVM,选择最佳者作为
"传统方法" 的代表,与后续 BERT 微调做对比实验。

运行: python src/sentiment/baseline.py
"""

import os
import json
import sys
import time
import joblib

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report, confusion_matrix)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

LABELS = ["negative", "neutral", "positive"]
LABEL_ZH = {"negative": "负面", "neutral": "中性", "positive": "正面"}

# 内置常用中文停用词表
STOPWORDS = set("""
的 了 和 是 在 我 你 他 她 它 们 这 那 有 也 都 就 不 而 及 与 着 或 又 很 太 更 最 还
一个 没有 我们 你们 他们 这个 那个 什么 怎么 为什么 因为 所以 但是 如果 虽然 然而 不过
还是 就是 可以 没有 非常 自己 已经 应该 需要 觉得 知道 时候 现在 这样 那样 之后 之前
一下 一点 一些 什么 东西 还是 不会 不能 不是 不要 这个 这些 那些 自己 就是 只是
""".split())


def load_data(split: str):
    with open(os.path.join(PROC_DIR, f"sentiment_{split}.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str):
    """jieba 分词 + 停用词过滤"""
    words = jieba.lcut(text)
    return [w.strip() for w in words if w.strip() and w.strip() not in STOPWORDS]


def main():
    print("=" * 60)
    print("情感分析 Baseline: TF-IDF + 传统分类器")
    print("=" * 60)

    train = load_data("train")
    val = load_data("validation")
    test = load_data("test")
    print(f"数据量: train={len(train)}  val={len(val)}  test={len(test)}")

    # 分词(空格连接,供 TfidfVectorizer 使用)
    print("jieba 分词中...")
    t0 = time.time()
    X_train_raw = [" ".join(tokenize(d["text"])) for d in train]
    X_test_raw = [" ".join(tokenize(d["text"])) for d in test]
    y_train = [d["label"] for d in train]
    y_test = [d["label"] for d in test]
    print(f"分词耗时 {time.time() - t0:.1f}s")

    # TF-IDF 特征
    print("提取 TF-IDF 特征...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),      # unigram + bigram
        min_df=3,                # 过滤低频词
        max_df=0.9,              # 过滤高频词
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    print(f"特征维度: {X_train.shape}, 耗时 {time.time() - t0:.1f}s")

    # 训练并对比多个分类器
    classifiers = {
        "朴素贝叶斯 (NB)": MultinomialNB(),
        "逻辑回归 (LR)": LogisticRegression(max_iter=1000, n_jobs=-1),
        "线性 SVM": LinearSVC(),
    }

    results = {}
    best_model, best_f1, best_name = None, -1, ""
    for name, clf in classifiers.items():
        print(f"\n--- 训练 {name} ---")
        t0 = time.time()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        print(f"  {name}: acc={acc:.4f}  macro-F1={f1:.4f}  (训练 {time.time() - t0:.1f}s)")
        results[name] = {"acc": acc, "f1": f1}
        if f1 > best_f1:
            best_f1, best_name, best_model = f1, name, clf

    print(f"\n最佳传统模型: {best_name} (macro-F1={best_f1:.4f})")

    # 详细评估最佳模型
    y_pred = best_model.predict(X_test)
    print("\n" + "=" * 60)
    print(f"[{best_name}] 详细评估")
    print("=" * 60)
    print(classification_report(
        y_test, y_pred,
        target_names=[LABEL_ZH[l] for l in LABELS],
        digits=4,
    ))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    print("混淆矩阵 (行=真实, 列=预测):")
    header = "          " + "".join(f"{LABEL_ZH[l]:>8}" for l in LABELS)
    print(header)
    for i, l in enumerate(LABELS):
        print(f"{LABEL_ZH[l]:>6}    " + "".join(f"{cm[i][j]:>8}" for j in range(len(LABELS))))

    # 保存最佳模型
    joblib.dump(best_model, os.path.join(MODEL_DIR, "sentiment_baseline.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    print(f"\n模型已保存: models/sentiment_baseline.pkl, models/tfidf_vectorizer.pkl")

    # 保存评估结果供后续对比实验使用
    metric = {
        "model": best_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
    }
    with open(os.path.join(MODEL_DIR, "baseline_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metric, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
