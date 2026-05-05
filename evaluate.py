import re
import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer  # 修改点1：用 rouge-score

# ===================== 【你只需要改这里的4个路径】 =====================
BASE_MODEL_PATH = "/home/NLP/CMQA-LIA/qwen2-medical"          # 基座模型路径
LORA_MODEL_PATH = "/home/NLP/CMQA-LIA/qwen_lora_final"         # LoRA 微调后权重路径
QUESTION_PATH = "/home/NLP/CMQA-LIA/data/cMedQA2/question.csv" # 你的问题CSV
ANSWER_PATH = "/home/NLP/CMQA-LIA/data/cMedQA2/answer.csv"     # 你的答案CSV
RAG_RETRIEVAL_FUNC_PATH = "/home/NLP/CMQA-LIA/rag.py"          # 你的 RAG 检索文件
# =====================================================================

# 量化配置（和你训练时一致）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 加载模型
def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = model.eval()
    return tokenizer, model

def load_lora_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
    model = model.eval()
    return tokenizer, model

# 加载 RAG（从你的 rag.py 导入）
def load_rag_model():
    tokenizer, lora_model = load_lora_model()
    from rag import retrieve_knowledge  # 你自己写的 RAG 检索函数
    return tokenizer, lora_model, retrieve_knowledge

# 生成回答（统一接口）
def generate_answer(tokenizer, model, question, max_new_tokens=512):
    inputs = tokenizer(question, return_tensors="pt")
    # 获取模型所在的设备，并将输入移动到相同设备
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=1.0,
            temperature=0.1,
            repetition_penalty=1.05
        )
    answer = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return answer.strip()

# RAG 组合生成
def generate_rag_answer(tokenizer, model, retrieve_func, question):
    context = retrieve_func(question)
    prompt = f"参考资料：{context}\n用户问题：{question}\n请根据参考资料回答："
    return generate_answer(tokenizer, model, prompt)

# ===================== 指标计算 =====================
# 1. 准确率（精确匹配）
def compute_acc(pred, ref):
    return 1 if pred.strip() == ref.strip() else 0

# 2. F1（词级别）
def compute_f1(pred, ref):
    pred_tokens = set(re.split(r'\W+', pred.lower()))
    ref_tokens = set(re.split(r'\W+', ref.lower()))
    common = pred_tokens & ref_tokens
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)

# 3. BLEU-4
smooth = SmoothingFunction()
def compute_bleu(pred, ref):
    pred_tokens = pred.split()
    ref_tokens = [ref.split()]
    try:
        return sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
    except:
        return 0.0

# 4. ROUGE-L（修改点2：用 rouge-score）
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
def compute_rouge_l(pred, ref):
    try:
        scores = scorer.score(ref, pred)
        return scores['rougeL'].fmeasure
    except:
        return 0.0

# ===================== 统一评测 =====================
def evaluate_model(model_name, tokenizer, model_func, test_data):
    print(f"\n========== 开始评测：{model_name} ==========")
    acc_list = []
    f1_list = []
    bleu_list = []
    rouge_list = []

    for data in tqdm(test_data):
        q = data["question"]
        ref = data["answer"]

        # 生成预测
        if model_name == "base":
            pred = generate_answer(tokenizer, model_func, q)
        elif model_name == "lora":
            pred = generate_answer(tokenizer, model_func, q)
        elif model_name == "lora_rag":
            tokenizer, model, retrieve = model_func
            pred = generate_rag_answer(tokenizer, model, retrieve, q)
        else:
            pred = ""

        # 计算指标
        acc = compute_acc(pred, ref)
        f1 = compute_f1(pred, ref)
        bleu = compute_bleu(pred, ref)
        rouge = compute_rouge_l(pred, ref)

        acc_list.append(acc)
        f1_list.append(f1)
        bleu_list.append(bleu)
        rouge_list.append(rouge)

    # 输出结果
    result = {
        "model": model_name,
        "Acc": round(np.mean(acc_list), 4),
        "F1": round(np.mean(f1_list), 4),
        "BLEU-4": round(np.mean(bleu_list), 4),
        "ROUGE-L": round(np.mean(rouge_list), 4)
    }
    print(result)
    return result

# ===================== 主函数 =====================
if __name__ == "__main__":
    # 1. 加载你的 cMedQA2 测试集
    df_question = pd.read_csv(QUESTION_PATH)
    df_answer = pd.read_csv(ANSWER_PATH)

    print("=== question.csv 列名 ===")
    print(df_question.columns.tolist())
    print("\n=== answer.csv 列名 ===")
    print(df_answer.columns.tolist())

    # 关键：用 question_id 合并两个表
    df_test = pd.merge(
        df_question, 
        df_answer, 
        on="question_id",  # 两个表都有的主键
        how="inner"
    )

    test_data = []
    for _, row in df_test.iterrows():
        test_data.append({
            "question": row["content_x"],  # question表的content（问题文本）
            "answer": row["content_y"]    # answer表的content（答案文本）
        })

    # 2. 评测基座模型
    tokenizer_base, model_base = load_base_model()
    res_base = evaluate_model("base", tokenizer_base, model_base, test_data)

    # 3. 评测 LoRA 模型
    tokenizer_lora, model_lora = load_lora_model()
    res_lora = evaluate_model("lora", tokenizer_lora, model_lora, test_data)

    # 4. 评测 LoRA + RAG
    rag_tokenizer, rag_model, rag_retrieve = load_rag_model()
    res_rag = evaluate_model("lora_rag", (rag_tokenizer, rag_model, rag_retrieve), None, test_data)

    # 5. 输出最终对比表格
    print("\n==================== 最终实验结果 ====================")
    print(f"模型\t\tAcc\tF1\tBLEU-4\tROUGE-L")
    print(f"基座模型\t{res_base['Acc']}\t{res_base['F1']}\t{res_base['BLEU-4']}\t{res_base['ROUGE-L']}")
    print(f"LoRA 微调\t{res_lora['Acc']}\t{res_lora['F1']}\t{res_lora['BLEU-4']}\t{res_lora['ROUGE-L']}")
    print(f"LoRA + RAG\t{res_rag['Acc']}\t{res_rag['F1']}\t{res_rag['BLEU-4']}\t{res_rag['ROUGE-L']}")

    # 保存结果
    with open("/home/NLP/CMQA-LIA/experiment_result.json", "w", encoding="utf-8") as f:
        json.dump([res_base, res_lora, res_rag], f, ensure_ascii=False, indent=2)