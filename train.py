import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    Qwen2ForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

# ===================== 核心修复区域 =====================
# 修复1: 强制指定可见设备为 0。
# 这一步至关重要，它让程序“以为”系统里只有一张卡，从而完全绕过 NCCL 的多卡通信逻辑。
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 修复2: 禁用 NCCL 的网络通信（InfiniBand），防止网络接口冲突导致的报错
os.environ["NCCL_IB_DISABLE"] = "1"
# 修复3: 指定 socket 接口为 lo (localhost)，防止因网卡选择错误导致的 NCCL Error 2
os.environ["NCCL_SOCKET_IFNAME"] = "lo"
# =======================================================

# ===================== 配置项 =====================
BASE_MODEL_PATH = "/home/NLP/CMQA-LIA/qwen2-medical"
DATASET_PATH = "/home/NLP/CMQA-LIA/data/cMedQA2"
LORA_SAVE_DIR = "/home/NLP/CMQA-LIA/qwen_lora_final"

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
# ===================================================

def main():
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # 加载4bit量化模型
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 注意：因为上面设置了 CUDA_VISIBLE_DEVICES="0"，这里直接用 "cuda" 即可，
    # 不需要写死 "cuda:0"，这样代码兼容性更好。
    model = Qwen2ForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=quantization_config,
        device_map="cuda"
    )

    # 注入LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 加载数据集
    def load_data():
        q = pd.read_csv(f"{DATASET_PATH}/question.csv")
        a = pd.read_csv(f"{DATASET_PATH}/answer.csv")
        df = pd.merge(q, a, on="question_id")
        df = df.rename(columns={'content_x': 'question', 'content_y': 'answer'})
        df = df.dropna(subset=["question", "answer"])
        df["input"] = "相关知识：无\n\n问题：" + df["question"]
        df["output"] = df["answer"]
        return Dataset.from_pandas(df[["input", "output"]])

    train_ds = load_data()

    # 编码
    def encode(examples):
        # 开启 truncation 和 padding
        inputs = tokenizer(examples["input"], max_length=512, truncation=True, padding="max_length")
        # 处理标签
        labels = tokenizer(examples["output"], max_length=512, truncation=True, padding="max_length")["input_ids"]
        # 将 pad_token 替换为 -100，这样计算 loss 时会忽略 padding 部分
        labels = [[-100 if x == tokenizer.pad_token_id else x for x in lab] for lab in labels]
        inputs["labels"] = labels
        return inputs

    # 使用 map 进行批处理，num_proc=1 防止多进程读取数据导致显存泄漏或死锁
    train_ds = train_ds.map(encode, batched=True, num_proc=1)

    # 训练参数
    args = TrainingArguments(
        output_dir=LORA_SAVE_DIR,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=10,
        # 确保你的显卡支持 bf16 (Volta架构及以上，如 T4, V100, 3090, 4090)
        # 如果显卡较老（如 2080Ti），请将 bf16=False 改为 fp16=True
        bf16=True,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        # 修复4: 移除所有 ddp_backend 设置，让 Trainer 自动检测单卡环境
        # 修复5: 显式禁用 deepspeed 等分布式策略（如果不需要的话）
        deepspeed=None,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds)
    trainer.train()

    # 保存
    model.save_pretrained(LORA_SAVE_DIR)
    tokenizer.save_pretrained(LORA_SAVE_DIR)
    print("✅ LoRA 训练完成，模型已保存到：", LORA_SAVE_DIR)

if __name__ == "__main__":
    main()