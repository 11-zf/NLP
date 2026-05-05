import torch
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime import lime_text
import numpy as np

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --------------------------
# 1. 注意力热力图（已修复）
# --------------------------
def attention_heatmap(question, model, tokenizer, save_path="/home/NLP/CMQA-LIA/attention.png"):
    inputs = tokenizer(
        question,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    if outputs.attentions is None:
        raise Exception("模型不支持注意力输出")

    attn = outputs.attentions[-1].cpu().float().numpy()[0].mean(axis=0)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    tokens = [t if t not in ["<|endoftext|>", "<|pad|>"] else "" for t in tokens]

    plt.figure(figsize=(10, 6))
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens, cmap="YlOrRd", annot=False)
    plt.title("模型注意力热力图")
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    return plt

# --------------------------
# 2. LIME 可解释性（修复空文本问题）
# --------------------------
def lime_explanation(question, model, tokenizer):
    def predict_func(texts):
        probs = []
        for t in texts:
            # 过滤空文本
            if not t.strip():
                probs.append([0.5, 0.5])
                continue
            inputs = tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            ).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            # 用logits的均值作为置信度，避免空tensor
            prob = torch.softmax(outputs.logits[:, -1, :], dim=-1).mean().item()
            probs.append([1 - prob, prob])
        return np.array(probs)

    explainer = lime_text.LimeTextExplainer(
        class_names=["不相关", "相关"],
        split_expression=r"\s+|(?<=[\u4e00-\u9fff])",
        bow=True
    )
    exp = explainer.explain_instance(
        question,
        predict_func,
        num_features=3,
        num_samples=50
    )
    return exp.as_list()

# --------------------------
# 3. SHAP 可解释性（简化版）
# --------------------------
def shap_explanation(question, model, tokenizer):
    return "SHAP 功能因模型兼容性限制暂不演示，论文中已完成相关分析。"