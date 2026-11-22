# -*- coding: utf-8 -*-
"""
情感分析:bert-base-chinese 微调(三分类)
=========================================
使用 HuggingFace Transformers + GPU 微调中文 BERT 做情感三分类。
通过 HF 国内镜像 (hf-mirror.com) 下载预训练模型。

运行: python src/sentiment/bert_train.py
"""

import os
import json
import sys
import time
import numpy as np

# 必须先设置镜像,再导入 transformers 相关
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "sentiment_bert")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_NAME = "bert-base-chinese"
MAX_LENGTH = 128
LABELS = ["negative", "neutral", "positive"]
LABEL_ZH = {"negative": "负面", "neutral": "中性", "positive": "正面"}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def load_data(split: str):
    with open(os.path.join(PROC_DIR, f"sentiment_{split}.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1}


def main():
    print("=" * 60)
    print(f"情感分析 BERT 微调: {MODEL_NAME}")
    print("=" * 60)
    print(f"GPU 可用: {torch.cuda.is_available()}  ->  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    train = load_data("train")
    test = load_data("test")

    # 转换为 Dataset
    train_ds = Dataset.from_dict({
        "text": [d["text"] for d in train],
        "label": [LABEL2ID[d["label"]] for d in train],
    })
    test_ds = Dataset.from_dict({
        "text": [d["text"] for d in test],
        "label": [LABEL2ID[d["label"]] for d in test],
    })

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

    print("分词中...")
    train_ds = train_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # 模型
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        eval_strategy="no",                # 训练中不评估,最后统一在 test 上评估
        save_strategy="epoch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=500,
        logging_steps=100,
        save_total_limit=1,
        load_best_model_at_end=False,
        fp16=torch.cuda.is_available(),    # GPU 上启用混合精度加速
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\n开始训练...")
    t0 = time.time()
    trainer.train()
    print(f"训练完成,耗时 {time.time() - t0:.1f}s")

    # 保存模型
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"模型已保存: {MODEL_DIR}")

    # 在 test 集上评估
    print("\n在测试集上评估...")
    preds = trainer.predict(test_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = test_ds["label"]

    print("\n" + "=" * 60)
    print("[BERT 微调] 测试集详细评估")
    print("=" * 60)
    print(classification_report(
        y_true, y_pred,
        target_names=[LABEL_ZH[l] for l in LABELS],
        digits=4,
    ))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)))
    print("混淆矩阵 (行=真实, 列=预测):")
    header = "          " + "".join(f"{LABEL_ZH[l]:>8}" for l in LABELS)
    print(header)
    for i, l in enumerate(LABELS):
        print(f"{LABEL_ZH[l]:>6}    " + "".join(f"{cm[i][j]:>8}" for j in range(len(LABELS))))

    # 保存评估指标
    metric = {
        "model": MODEL_NAME,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metric, f, ensure_ascii=False, indent=2)
    print(f"\n评估指标已保存: {MODEL_DIR}/metrics.json")


if __name__ == "__main__":
    main()
