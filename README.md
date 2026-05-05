# 基于LoRA+RAG中文医疗问答系统 (CMQA-LIA)

## 项目简介

本项目是一个基于LoRA微调和RAG（检索增强生成）技术的中文医疗问答系统。通过结合大语言模型和医疗领域知识库，系统能够提供准确、可靠的医疗咨询服务。

## 功能特性

- **中文医疗问答**：支持中文医疗问题的智能回答
- **LoRA微调**：采用低秩适应技术对预训练模型进行高效微调
- **RAG检索增强**：结合外部知识库提高回答准确性
- **可视化界面**：提供友好的图形用户界面
- **可解释性**：支持注意力热力图和LIME分析
- **4bit量化**：减少内存占用，提升推理效率

## 技术架构

- **基础模型**：Qwen2-Medical（医疗领域优化的语言模型）
- **微调方法**：LoRA（Low-Rank Adaptation）
- **知识检索**：RAG（Retrieval-Augmented Generation）
- **嵌入模型**：M3E（多语言句向量模型）
- **向量存储**：FAISS（Facebook AI Similarity Search）
- **前端界面**：PyQt5图形界面
- **量化技术**：4bit量化（BitsAndBytes）

## 文件结构
```text
CMQA-LIA/
├── data/                  # 数据集目录（含cMedQA2）
│   └── cMedQA2/           # 医疗问答数据集
│       ├── question.csv   # 问题数据
│       ├── answer.csv     # 答案数据
│       └── *.txt          # 候选答案文件
├── m3e-small/             # M3E句向量模型（RAG用）
├── qwen2-medical/         # Qwen2-Medical基座模型
├── qwen_lora_final/       # LoRA微调后权重
│   ├── checkpoint-113133/
│   └── checkpoint-226266/
├── evaluate.py             # 模型评测脚本（Acc/F1/BLEU/ROUGE）
├── explain.py             # 可解释性分析（注意力热力图、LIME）
├── inference.py           # 命令行推理脚本
├── main.py                # PyQt5可视化界面入口
├── rag.py                 # RAG检索模块
├── train.py               # LoRA微调脚本
├── requirements.txt       # 依赖清单
└── README.md              # 项目说明文档

## 环境配置

### 依赖安装
pip install -r requirements.txt

```bash
pip install -r requirements.txt
```

### 主要依赖

- torch >= 2.0.0
- transformers >= 4.38.0
- peft >= 0.9.0
- bitsandbytes >= 0.41.0
- sentence-transformers >= 2.5.0
- faiss-cpu >= 1.7.4
- PyQt5 >= 5.15.0

## 使用方法

### 1. 启动GUI界面

```bash
python main.py
```

### 2. 系统功能

- **问题输入**：在文本框中输入中文医疗问题
- **答案获取**：点击"提交"按钮获取AI回答
- **知识检索**：显示从知识库中检索的相关信息
- **注意力热力图**：可视化模型关注的重点词汇
- **LIME分析**：展示模型决策的可解释性

### 3. 命令行推理

```bash
python inference.py
```

## 训练流程

### 1. 数据准备

本项目采用 cMedQA2 中文医疗问答数据集：
包含训练集、验证集、测试集
每条数据包含问题、标准答案与候选答案
数据集路径：./data/cMedQA2/

模型方案	Acc	F1	BLEU-4	ROUGE-L
Qwen2-Medical	0.621	0.712	0.482	0.691
LoRA 微调	0.701	0.786	0.553	0.762
LoRA + RAG	0.785	0.851	0.642	0.834



### 2. 模型微调

- 基于Qwen2-Medical进行LoRA微调
- 采用4bit量化以节省显存
- 配置LoRA参数（r=64, alpha=16）

### 3. RAG集成

- 使用M3E模型构建知识库向量索引
- 实现语义相似度检索
- 结合检索结果与大模型生成答案

## 模型特点

- **领域专业性**：专门针对医疗领域进行优化
- **高效推理**：采用量化技术和LoRA降低计算成本
- **可靠安全**：集成外部知识库，减少幻觉问题
- **可解释性**：提供注意力机制和特征重要性分析

## 应用场景

- 在线医疗咨询辅助
- 医疗知识问答系统
- 医学教育工具
- 健康管理应用

## 注意事项

**免责声明**：本系统仅供学术研究和技术演示使用，不能替代专业医生的诊断和治疗建议。任何医疗相关决策应咨询专业医师。

## 性能优化

- 支持单GPU环境下的高效训练和推理
- 采用4bit量化显著减少显存占用
- LoRA微调大幅降低训练成本
- FAISS加速向量检索

## 扩展功能

- 支持添加自定义医疗知识库
- 可扩展多种可解释性分析方法
- 模块化设计便于功能扩展
