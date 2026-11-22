// 生成实验报告 docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType, VerticalAlign, ImageRun, LevelFormat,
} = require("docx");

const FONT = "Microsoft YaHei";
const REPORT_DIR = __dirname;

// ---- 辅助函数 ----
function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text, bold: true })] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text, bold: true })] });
}
function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120, line: 360 },
    children: [new TextRun({ text, ...opts })],
  });
}
function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bl", level: 0 },
    spacing: { after: 60 },
    children: [new TextRun({ text })],
  });
}

// 表格边框
const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const cellBorders = { top: border, bottom: border, left: border, right: border };

function makeTable(headers, rows, colWidths) {
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h) =>
      new TableCell({
        borders: cellBorders,
        width: { size: colWidths[headers.indexOf(h)], type: WidthType.DXA },
        shading: { fill: "D5E8F0", type: ShadingType.CLEAR },
        verticalAlign: VerticalAlign.CENTER,
        children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: h, bold: true })] })],
      })
    ),
  });
  const dataRows = rows.map((row) =>
    new TableRow({
      children: row.map((cell, i) =>
        new TableCell({
          borders: cellBorders,
          width: { size: colWidths[i], type: WidthType.DXA },
          verticalAlign: VerticalAlign.CENTER,
          children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: cell })] })],
        })
      ),
    })
  );
  return new Table({ columnWidths: colWidths, margins: { top: 80, bottom: 80, left: 120, right: 120 }, rows: [headerRow, ...dataRows] });
}

// 比较图(如果存在)
let comparisonImg = [];
const cmpPath = path.join(REPORT_DIR, "comparison.png");
if (fs.existsSync(cmpPath)) {
  comparisonImg = [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 120 },
      children: [new ImageRun({
        type: "png",
        data: fs.readFileSync(cmpPath),
        transformation: { width: 480, height: 300 },
        altText: { title: "对比图", description: "情感分析对比", name: "comparison" },
      })],
    }),
  ];
}

const children = [
  // 标题
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text: "基于预训练模型的中文文本情感分析", bold: true, size: 36 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
    children: [new TextRun({ text: "与信息抽取系统设计 —— 实验报告", bold: true, size: 36 })],
  }),

  // 摘要
  h2("摘要"),
  p("本实验设计并实现了一个完整的中文文本情感分析与信息抽取系统。系统包含文本预处理、情感三分类(正面/负面/中性)、命名实体识别、结果可视化四大模块,并对比了传统机器学习方法与预训练模型的效果。情感分析方面,使用 TF-IDF + 逻辑回归作为传统 baseline,使用 bert-base-chinese 微调作为改进方法;信息抽取方面,使用 BERT 序列标注(BIO)对 CLUENER2020 数据集进行命名实体识别。实验结果表明,预训练模型 BERT 在两项任务上均显著优于传统方法,验证了预训练语言模型在中文 NLP 任务中的有效性。"),

  // 一、实验目的
  h1("一、实验目的"),
  bullet("掌握中文文本预处理流程(分词、去噪、标准化)。"),
  bullet("理解情感分析、命名实体识别的基本方法。"),
  bullet("能够设计并实现一个完整的 NLP 小系统。"),
  bullet("能够对比传统方法与预训练模型的效果。"),
  bullet("能够撰写包含实验设计、结果分析和错误案例的实验报告。"),

  // 二、实验环境
  h1("二、实验环境"),
  makeTable(
    ["项目", "配置"],
    [
      ["操作系统", "Windows 11"],
      ["编程语言", "Python 3.14"],
      ["深度学习框架", "PyTorch 2.11.0 (CUDA 12.8)"],
      ["GPU", "NVIDIA GeForce RTX 4060 Laptop (8GB)"],
      ["预训练模型", "bert-base-chinese"],
      ["主要库", "transformers 5.24, jieba, scikit-learn, streamlit"],
    ],
    [3000, 6026]
  ),

  // 三、数据集
  h1("三、数据集"),
  h2("3.1 情感分析数据集"),
  p("情感分析采用 amazon_reviews_multi 多语言亚马逊评论数据集的中文部分。该数据集每条评论包含 1-5 星评分,本实验将星级映射为三分类标签:1-2 星为负面、3 星为中性、4-5 星为正面。原始训练集 20 万条(各星级均衡,每星 4 万条),本实验对每个类别随机采样 2 万条,共 6 万条用于训练,并保留 5000 条验证集与 5000 条测试集。"),
  h2("3.2 信息抽取数据集"),
  p("命名实体识别采用 CLUENER2020 数据集,该数据集是中文细粒度命名实体识别基准,包含 10 类实体:人名、地名、机构名、公司、游戏、电影、书籍、政府、职位、场景。训练集 10748 条,验证集 1343 条,测试集 1345 条。"),

  // 四、文本预处理
  h1("四、文本预处理"),
  bullet("去噪:使用正则表达式去除 HTML 标签、URL 链接及特殊符号,仅保留中文、字母、数字与常用标点。"),
  bullet("标准化:合并多余空白字符,统一文本格式。"),
  bullet("分词:使用 jieba 进行中文分词(仅传统方法需要,BERT 使用字符级 tokenizer)。"),
  bullet("去重与过滤:去除重复文本及长度过短的样本。"),
  bullet("NER 标注转换:将 CLUENER2020 的实体标注转换为 BIO 序列标注格式。"),

  // 五、情感分析实验
  h1("五、情感分析实验"),
  h2("5.1 Baseline:TF-IDF + 传统分类器"),
  p("baseline 使用 jieba 分词 + TF-IDF 特征(ngram 1-2),分别训练朴素贝叶斯、逻辑回归、线性 SVM 三种分类器,在测试集上的结果如下:"),
  makeTable(
    ["分类器", "准确率", "宏平均 F1"],
    [
      ["朴素贝叶斯", "0.6272", "0.5966"],
      ["逻辑回归", "0.6488", "0.6232"],
      ["线性 SVM", "0.6202", "0.5936"],
    ],
    [3000, 3000, 3026]
  ),
  p("其中逻辑回归效果最好,宏平均 F1 为 0.6232,作为传统方法的代表。"),
  h2("5.2 BERT 微调"),
  p("使用 bert-base-chinese 预训练模型进行微调,在分类层输出三分类。训练参数:batch_size=32,学习率 2e-5,训练 3 个 epoch,最大序列长度 128,使用 GPU 混合精度训练。"),
  h2("5.3 对比实验结果"),
  makeTable(
    ["方法", "准确率", "宏平均 F1"],
    [
      ["TF-IDF + 逻辑回归", "0.6488", "0.6232"],
      ["bert-base-chinese 微调", "0.7062", "0.6813"],
    ],
    [3000, 3000, 3026]
  ),
  p("BERT 微调相比传统方法准确率提升 5.74 个百分点,宏平均 F1 提升 5.80 个百分点,验证了预训练模型的优势。"),
  ...comparisonImg,

  // 六、信息抽取实验
  h1("六、信息抽取(命名实体识别)实验"),
  p("使用 bert-base-chinese 微调进行序列标注,采用 BIO 标注体系(共 21 个标签)。训练参数:batch_size=32,学习率 3e-5,训练 5 个 epoch。使用 seqeval 进行实体级评估,结果如下:"),
  makeTable(
    ["实体类型", "精确率", "召回率", "F1"],
    [
      ["人名", "0.8496", "0.8989", "0.8736"],
      ["电影", "0.8344", "0.8344", "0.8344"],
      ["游戏", "0.7798", "0.8644", "0.8199"],
      ["政府", "0.7609", "0.8502", "0.8031"],
      ["书籍", "0.7590", "0.8182", "0.7875"],
      ["职位", "0.7426", "0.8129", "0.7762"],
      ["公司", "0.7245", "0.8069", "0.7635"],
      ["机构", "0.7229", "0.7820", "0.7513"],
      ["场景", "0.6789", "0.7081", "0.6932"],
      ["地名", "0.5742", "0.6434", "0.6068"],
      ["微平均", "0.7386", "0.8031", "0.7695"],
      ["宏平均", "0.7427", "0.8020", "0.7709"],
    ],
    [2200, 2200, 2200, 2426]
  ),
  p("整体宏平均 F1 为 0.7709,其中人名识别效果最好(F1=0.8736),地名识别效果较差(F1=0.6068),主要原因在于地名边界模糊且表达形式多样。"),

  // 七、错误案例分析
  h1("七、错误案例分析"),
  p("对 BERT 与 baseline 在测试集上的预测结果进行对比分析,共 5000 条测试样本:"),
  makeTable(
    ["错误类型", "数量", "说明"],
    [
      ["两者都判错", "1054", "困难样本,两模型均无法正确分类"],
      ["baseline 错、BERT 对", "702", "BERT 纠正了 baseline 的错误"],
      ["BERT 错、baseline 对", "416", "BERT 在某些样本上表现不如 baseline"],
    ],
    [3000, 2000, 4026]
  ),
  p("典型错误案例:如\"强烈建议不要购买 因为里面的知识太过时了\"(真实:负面),baseline 判为中性而 BERT 正确判为负面,说明 BERT 能更好地理解隐含的情感倾向。而\"总体不错。裤型是喜欢的,但两个裤腿长度不一样\"这类正负情感混杂的文本,两模型均判错,反映了混合情感表达的识别仍是难点。"),

  // 八、系统可视化
  h1("八、系统可视化界面"),
  p("基于 Streamlit 构建了交互式 Web 界面,主要功能包括:"),
  bullet("单条文本分析:输入文本,实时返回情感倾向及置信度,并高亮显示识别出的实体。"),
  bullet("批量文本分析:对多条文本进行情感分布统计、词云生成、实体类型统计。"),
  bullet("模型切换:支持在传统方法与 BERT 微调之间切换,便于对比。"),

  // 九、总结与展望
  h1("九、总结与展望"),
  p("本实验完成了一个完整的中文 NLP 系统,对比了传统机器学习方法与预训练模型在情感分析和命名实体识别任务上的表现。实验表明,BERT 预训练模型通过迁移学习显著提升了中文文本理解能力。未来的改进方向包括:引入更大规模的预训练模型(RoBERTa)、使用 CRF 层提升序列标注的一致性、扩充中性情感样本以缓解类别不平衡,以及引入方面级情感分析等更细粒度的任务。"),
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: "1F4E79", font: FONT },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, color: "2E74B5", font: FONT },
        paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bl",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } },
    },
    children,
  }],
});

const outPath = path.join(REPORT_DIR, "实验报告.docx");
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(outPath, buf);
  console.log("报告已生成:", outPath);
});
