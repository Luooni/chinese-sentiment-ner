# -*- coding: utf-8 -*-
"""
中文文本情感分析与信息抽取系统 - Streamlit 交互界面
====================================================
功能:
1. 单条文本:情感三分类预测(传统/BERT 可选)+ 命名实体抽取
2. 批量分析:情感分布、词云、实体类型统计可视化

运行: streamlit run app.py
"""

import os
import sys
import json
import re

import numpy as np
import jieba
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st

# 设置 matplotlib 中文字体(解决图表中文乱码/不显示问题)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
STOPWORDS = set("""
的 了 和 是 在 我 你 他 她 它 们 这 那 有 也 都 就 不 而 及 与 着 或 又 很 太 更 最 还
一个 没有 我们 你们 他们 这个 那个 什么 怎么 为什么 因为 所以 但是 如果 虽然 然而 不过
还是 就是 可以 没有 非常 自己 已经 应该 需要 觉得 知道 时候 现在 这样 那样 之后 之前
一下 一点 一些 什么 东西 还是 不会 不能 不是 不要 这个 这些 那些 自己 就是 只是
""".split())

LABELS = ["negative", "neutral", "positive"]
LABEL_ZH = {"negative": "负面", "neutral": "中性", "positive": "正面"}
LABEL_COLOR = {"negative": "#E34D59", "neutral": "#F5A623", "positive": "#34A853"}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

NER_TYPES = ["name", "address", "organization", "company", "game",
             "movie", "book", "government", "position", "scene"]
NER_TYPE_ZH = {
    "name": "人名", "address": "地名", "organization": "机构",
    "company": "公司", "game": "游戏", "movie": "电影",
    "book": "书籍", "government": "政府", "position": "职位", "scene": "场景",
}
NER_LABELS = ["O"] + [f"{p}-{t}" for t in NER_TYPES for p in ("B", "I")]
NER_ID2LABEL = {i: l for i, l in enumerate(NER_LABELS)}


# ---------------- 模型加载(缓存) ----------------
@st.cache_resource
def load_sentiment_baseline():
    clf = joblib.load(os.path.join(MODEL_DIR, "sentiment_baseline.pkl"))
    vec = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    return clf, vec


@st.cache_resource
def load_sentiment_bert():
    path = os.path.join(MODEL_DIR, "sentiment_bert")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model


@st.cache_resource
def load_ner_bert():
    path = os.path.join(MODEL_DIR, "ner_bert")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForTokenClassification.from_pretrained(path)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model


# ---------------- 预测函数 ----------------
def predict_sentiment_baseline(text):
    clf, vec = load_sentiment_baseline()
    words = [w for w in jieba.lcut(text) if w.strip() and w.strip() not in STOPWORDS]
    feat = vec.transform([" ".join(words)])
    proba = clf.predict_proba(feat)[0] if hasattr(clf, "predict_proba") else None
    pred = clf.predict(feat)[0]
    return pred, proba


def predict_sentiment_bert(text):
    tokenizer, model = load_sentiment_bert()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits
    proba = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    pred = ID2LABEL[int(np.argmax(proba))]
    return pred, proba


def bio_decode(chars, tags):
    """BIO 标签序列解码为实体列表"""
    entities = []
    cur = None
    for c, t in zip(chars, tags):
        if t.startswith("B-"):
            if cur:
                entities.append(cur)
            cur = {"type": t[2:], "text": c}
        elif t.startswith("I-") and cur and cur["type"] == t[2:]:
            cur["text"] += c
        else:
            if cur:
                entities.append(cur)
            cur = None
    if cur:
        entities.append(cur)
    return entities


def predict_ner(text):
    tokenizer, model = load_ner_bert()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    word_ids = inputs.word_ids()          # 先取 word_ids,避免后续转设备后丢失
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits
    preds = torch.argmax(logits, dim=-1)[0].cpu().numpy()

    char_tags = []
    prev = None
    for w_id, p_id in zip(word_ids, preds):
        if w_id is None:
            continue
        if w_id != prev:
            char_tags.append(NER_ID2LABEL[int(p_id)])
        prev = w_id

    entities = bio_decode(list(text), char_tags)
    return entities


# ---------------- 可视化 ----------------
def plot_wordcloud(texts):
    words = []
    for t in texts:
        words.extend([w for w in jieba.lcut(t) if len(w) > 1 and w not in STOPWORDS])
    text = " ".join(words)
    font_path = None
    for f in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
        if os.path.exists(f):
            font_path = f
            break
    wc = WordCloud(font_path=font_path, width=800, height=400,
                   background_color="white", max_words=100).generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


def plot_sentiment_dist(labels):
    from collections import Counter
    c = Counter(labels)
    # 只保留有值的类别,避免 0% 标签重叠
    items = [(LABEL_ZH[l], c.get(l, 0), LABEL_COLOR[l]) for l in LABELS if c.get(l, 0) > 0]
    if not items:
        items = [(LABEL_ZH[l], 0, LABEL_COLOR[l]) for l in LABELS]
    names = [i[0] for i in items]
    sizes = [i[1] for i in items]
    colors = [i[2] for i in items]

    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=names, colors=colors, autopct="%1.1f%%",
        startangle=90, labeldistance=1.15, pctdistance=0.7,
    )
    # 调整文字样式:类别名放外侧,百分比白色显示在色块上
    for t in texts:
        t.set_fontsize(10)
    for t in autotexts:
        t.set_color("white")
        t.set_fontsize(9)
        t.set_weight("bold")
    ax.set_title("情感分布")
    return fig


# ---------------- 界面 ----------------
st.set_page_config(page_title="中文情感分析与信息抽取系统", layout="wide")
st.title("🔍 中文文本情感分析与信息抽取系统")
st.markdown("基于预训练模型 BERT 的 NLP 系统 —— 情感三分类 + 命名实体识别")

tab1, tab2 = st.tabs(["📝 单条分析", "📊 批量分析"])

with tab1:
    st.subheader("输入文本")
    text = st.text_area("请输入中文评论/文本:", height=120,
                        value="这家餐厅味道不错，但上菜太慢，服务员态度也很差。")
    method = st.radio("情感分析方法:", ["BERT 微调(推荐)", "传统方法(TF-IDF+LR)"],
                      horizontal=True)

    if st.button("开始分析", type="primary"):
        if not text.strip():
            st.warning("请输入文本")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("😊 情感分析")
                if "BERT" in method:
                    pred, proba = predict_sentiment_bert(text)
                    label_names = [LABEL_ZH[l] for l in LABELS]
                else:
                    pred, proba = predict_sentiment_baseline(text)
                    label_names = [LABEL_ZH[l] for l in LABELS]
                st.markdown(f"**情感倾向: <span style='color:{LABEL_COLOR[pred]};font-size:22px'>{LABEL_ZH[pred]}</span>**",
                            unsafe_allow_html=True)
                if proba is not None:
                    for i, l in enumerate(LABELS):
                        st.progress(float(proba[i]) if hasattr(proba[i], "item") else float(proba[i]),
                                    text=f"{LABEL_ZH[l]}: {proba[i]:.2%}")
            with col2:
                st.subheader("🏷️ 信息抽取")
                entities = predict_ner(text)
                if not entities:
                    st.info("未抽取到实体")
                else:
                    st.markdown("**识别到的实体:**")
                    for e in entities:
                        st.markdown(f"- **{NER_TYPE_ZH.get(e['type'], e['type'])}**: {e['text']}")

with tab2:
    st.subheader("批量文本分析")
    batch_text = st.text_area("粘贴多条文本(每行一条):", height=150,
                              value="这家餐厅味道不错，但上菜太慢\n产品质量很好，值得购买\n太差了，用了两天就坏了")
    if st.button("批量分析", type="primary"):
        lines = [l.strip() for l in batch_text.split("\n") if l.strip()]
        if not lines:
            st.warning("请输入文本")
        else:
            labels = []
            all_entities = []
            with st.spinner("分析中..."):
                for line in lines:
                    pred, _ = predict_sentiment_bert(line)
                    labels.append(pred)
                    all_entities.extend(predict_ner(line))

            # 情感分布与实体统计并排显示
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("情感分布")
                st.pyplot(plot_sentiment_dist(labels))
            with c2:
                st.subheader("实体类型统计")
                from collections import Counter
                type_counter = Counter(e["type"] for e in all_entities)
                if type_counter:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    types = list(type_counter.keys())
                    counts = [type_counter[t] for t in types]
                    ax.bar([NER_TYPE_ZH.get(t, t) for t in types], counts, color="#5B8FF9")
                    ax.set_xlabel("实体类型")
                    ax.set_ylabel("数量")
                    ax.set_title("实体类型统计")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                else:
                    st.info("未抽取到实体")

            st.subheader("词云")
            st.pyplot(plot_wordcloud(lines))
