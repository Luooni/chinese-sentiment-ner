# -*- coding: utf-8 -*-
"""
命名实体识别:bert-base-chinese 微调(BIO 序列标注)
==================================================
基于 CLUENER2020 微调中文 BERT 做实体识别,识别 10 类实体:
人名/地名/机构/公司/游戏/电影/书籍/政府/职位/场景。

核心难点:将字符级 BIO 标签对齐到 BERT 的 subword token。

运行: python src/ner/bert_ner.py
"""

import os
import json
import sys
import time
import numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          TrainingArguments, Trainer)
from seqeval.metrics import classification_report as seq_report
from seqeval.metrics import f1_score as seq_f1
from seqeval.metrics import accuracy_score as seq_acc

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "ner_bert")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_NAME = "bert-base-chinese"
MAX_LENGTH = 128

NER_TYPES = [
    "name", "address", "organization", "company", "game",
    "movie", "book", "government", "position", "scene",
]

# 构建 BIO 标签体系: O + B-*/I-* 共 21 个
LABELS = ["O"] + [f"{p}-{t}" for t in NER_TYPES for p in ("B", "I")]
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
print(f"标签数: {len(LABELS)}  标签: {LABELS}")


def load_data(split: str):
    with open(os.path.join(PROC_DIR, f"ner_{split}.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize_and_align(examples, tokenizer):
    """分词并将字符级 BIO 标签对齐到 subword token"""
    tokenized = tokenizer(
        examples["chars"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        is_split_into_words=True,
    )
    labels = []
    for i, tags in enumerate(examples["tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:            # [CLS]/[SEP]/[PAD]
                label_ids.append(-100)
            elif word_idx != prev_word_idx:  # 每个 word 的第一个 subword
                label_ids.append(LABEL2ID[tags[word_idx]])
            else:                            # 后续 subword 忽略(中文按字切分,通常不会出现)
                label_ids.append(-100)
            prev_word_idx = word_idx
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_tags = []
    pred_tags = []
    for pred_row, label_row in zip(predictions, labels):
        true_seq = []
        pred_seq = []
        for p_id, l_id in zip(pred_row, label_row):
            if l_id != -100:   # 忽略 padding / 特殊 token
                true_seq.append(ID2LABEL[l_id])
                pred_seq.append(ID2LABEL[p_id])
        true_tags.append(true_seq)
        pred_tags.append(pred_seq)

    return {
        "f1": seq_f1(true_tags, pred_tags),
        "accuracy": seq_acc(true_tags, pred_tags),
    }


def main():
    print("=" * 60)
    print(f"NER BERT 微调: {MODEL_NAME}")
    print("=" * 60)
    print(f"GPU 可用: {torch.cuda.is_available()}")

    train = load_data("train")
    dev = load_data("dev")

    train_ds = Dataset.from_dict({
        "chars": [d["chars"] for d in train],
        "tags": [d["tags"] for d in train],
    })
    dev_ds = Dataset.from_dict({
        "chars": [d["chars"] for d in dev],
        "tags": [d["tags"] for d in dev],
    })

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("分词并对齐标签...")
    train_ds = train_ds.map(lambda x: tokenize_and_align(x, tokenizer), batched=True)
    dev_ds = dev_ds.map(lambda x: tokenize_and_align(x, tokenizer), batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=5,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_steps=150,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
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

    # 在 dev 集上最终评估
    print("\n在 dev 集上评估...")
    preds = trainer.predict(dev_ds)
    predictions = np.argmax(preds.predictions, axis=2)
    labels = preds.label_ids

    true_tags, pred_tags = [], []
    for pred_row, label_row in zip(predictions, labels):
        ts, ps = [], []
        for p_id, l_id in zip(pred_row, label_row):
            if l_id != -100:
                ts.append(ID2LABEL[l_id])
                ps.append(ID2LABEL[p_id])
        true_tags.append(ts)
        pred_tags.append(ps)

    print("\n" + "=" * 60)
    print("[NER] dev 集实体级评估 (seqeval)")
    print("=" * 60)
    print(seq_report(true_tags, pred_tags, digits=4))

    metric = {
        "model": MODEL_NAME,
        "entity_f1": float(seq_f1(true_tags, pred_tags)),
        "entity_accuracy": float(seq_acc(true_tags, pred_tags)),
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metric, f, ensure_ascii=False, indent=2)
    print(f"\n评估指标已保存: {MODEL_DIR}/metrics.json")


if __name__ == "__main__":
    main()
