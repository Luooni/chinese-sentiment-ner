# -*- coding: utf-8 -*-
"""
对比实验与错误案例分析
========================
汇总传统方法(Baseline)与 BERT 微调的评估结果,生成对比图表,
并对 BERT 的错误案例进行抽样分析。

运行: python src/evaluate.py
"""

import os
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 无界面后端,避免显示窗口
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "report")
os.makedirs(REPORT_DIR, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_metrics():
    """汇总并对比两种方法"""
    baseline = load_json(os.path.join(MODEL_DIR, "baseline_metrics.json"))
    bert = load_json(os.path.join(MODEL_DIR, "sentiment_bert", "metrics.json"))

    print("=" * 60)
    print("对比实验:传统方法 vs BERT 微调(情感分析)")
    print("=" * 60)
    print(f"{'方法':<24}{'准确率':>10}{'macro-F1':>10}")
    print("-" * 44)
    print(f"{baseline['model']:<24}{baseline['accuracy']:>10.4f}{baseline['macro_f1']:>10.4f}")
    print(f"{bert['model']:<24}{bert['accuracy']:>10.4f}{bert['macro_f1']:>10.4f}")
    print("-" * 44)
    acc_gain = (bert["accuracy"] - baseline["accuracy"]) * 100
    f1_gain = (bert["macro_f1"] - baseline["macro_f1"]) * 100
    print(f"BERT 相对提升:  准确率 +{acc_gain:.2f}pp, macro-F1 +{f1_gain:.2f}pp")

    # 生成对比柱状图
    methods = [baseline["model"], bert["model"]]
    accs = [baseline["accuracy"], bert["accuracy"]]
    f1s = [baseline["macro_f1"], bert["macro_f1"]]

    x = np.arange(2)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width / 2, accs, width, label="准确率 (Accuracy)", color="#5B8FF9")
    b2 = ax.bar(x + width / 2, f1s, width, label="宏平均 F1 (macro-F1)", color="#61C0BF")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("分数")
    ax.set_ylim(0, 1.05)
    ax.set_title("情感分析:传统方法 vs BERT 微调", fontsize=14)
    ax.legend()
    for b in (b1, b2):
        for rect in b:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.02, f"{h:.4f}",
                    ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    out = os.path.join(REPORT_DIR, "comparison.png")
    fig.savefig(out, dpi=150)
    print(f"\n对比图已保存: {out}")

    return baseline, bert


if __name__ == "__main__":
    compare_metrics()
