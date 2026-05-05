import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from rag import RAG
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="peft")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

BASE_MODEL_PATH = "/home/NLP/CMQA-LIA/qwen2-medical"
LORA_MODEL_PATH = "/home/NLP/CMQA-LIA/qwen_lora_final/checkpoint-226266"


# 4bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# RAG
rag = RAG()

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = Qwen2ForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
    attn_implementation="eager"
)
model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
model.eval()

# 修复版问答函数（关键！）
def medical_qa(question):
    knows = rag.retrieve(question)
    knows_str = "\n".join(knows)
    # 固定格式prompt，强制模型按格式回答
    prompt = f"""你是专业的医疗助手，请根据参考知识回答问题，回答要简洁、准确，不要生成无关内容。

参考知识：
{knows_str}

用户问题：{question}

回答："""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "回答：" in full_text:
        answer = full_text.split("回答：")[-1].strip()
    else:
        answer = full_text.strip()
    # 简单清洗乱码
    answer = answer.replace("。。。", "。").replace("，，，", "，")
    return answer, knows