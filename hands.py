from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from peft import PeftModel
from torch.cuda.amp import autocast
import gc
from collections import defaultdict
import torch.nn as nn
import os

DATA_DIR = "./data"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LORA_WEIGHTS_PATH = "./lora/model"
CLASSIFICATION_HEAD_PATH = "./lora/teacher_classification_head.pt"


# 释放显存
torch.cuda.empty_cache()

def read_jsonl(file_path):
    """读取 JSONL 文件并返回数据列表。"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file, start=1):
            stripped_line = line.strip()
            if stripped_line:  # 确保行非空
                try:
                    data.append(json.loads(stripped_line))
                except json.JSONDecodeError as e:
                    print(f"Failed to parse line {i}: {e}")
    return data

def clean_and_split(diseases_str, key_name):
    """清理并分割疾病名称字符串。"""
    if not isinstance(diseases_str, str):
        print(f"Warning: Expected string for {key_name}, got {type(diseases_str)} instead.")
        return []

    prefix = "Disease that the patient may acquire: "
    if diseases_str.startswith(prefix):
        diseases_str = diseases_str[len(prefix):]

    if not diseases_str.strip():
        return []

    return [d.strip() for d in diseases_str.split(",") if d.strip()]


def load_model_and_tokenizer(model_name, lora_weights_path, device):
    """加载模型和分词器。"""
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, lora_weights_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.to(device).eval()
    return model, tokenizer

# 定义分类模型
class LoRAClassificationModel(nn.Module):
    def __init__(self, base_model, classification_head):
        super().__init__()
        self.base_model = base_model
        self.classification_head = classification_head

    def forward(self, input_ids, **kwargs):  # 接受额外的参数
        # 获取base model的输出，包括隐藏层状态和logits
        outputs = self.base_model(input_ids=input_ids, output_hidden_states=True, **kwargs)
        # 使用最后一层隐藏状态进行分类
        last_token_hidden_state = outputs.hidden_states[-1][:, -1, :]  # 取最后一个 token 的隐藏状态
        last_token_hidden_state = last_token_hidden_state.to(self.classification_head.weight.dtype)
        # 通过分类头产生最终的logits
        logits = self.classification_head(last_token_hidden_state)
        return logits



def process_item(item, tokenizer, classification_model, device, labels):
    """处理单个数据项并返回标签概率。"""
    if "output" in item:
        label = clean_and_split(item["output"], "output")
    if "prompt" in item:
        instructions = item['prompt']
    if "predict" in item:
        predict = item["predict"]

    instruction_part = item.get("instruction", "")
    input_part = item.get("input", "")
    instructions = f"{instruction_part}\n{input_part}" if input_part else instruction_part

    # 使用模板生成输入文本
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": instructions}],
        tokenize=False, add_generation_prompt=True
    )

    # 将输入文本转换为模型输入
    inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=1024).to(device)

    # 使用 classification_model 进行推理
    with torch.no_grad():
        outputs = classification_model(**inputs)  # 使用分类模型
        logits = outputs  # classification_model 的输出已经是分类头的 logits

    # 计算每个标签的概率
    batch_sample_probs = {}
    probs = torch.sigmoid(logits)  # 将 logits 转换为概率
    for sample_idx in range(probs.shape[0]):  # 遍历 batch 中的每个样本
        batch_sample_probs[sample_idx] = {}
        for label_idx, label in enumerate(labels):
            batch_sample_probs[sample_idx][label] = probs[sample_idx, label_idx].item()

    # 释放显存
    del inputs, outputs, logits
    torch.cuda.empty_cache()
    gc.collect()

    return batch_sample_probs




def main():
    # 疾病标签（顺序与训练保持一致）
    labels = [
        "Acute and unspecified renal failure",
        "Acute cerebrovascular disease",
        "Acute myocardial infarction",
        "Cardiac dysrhythmias",
        "Chronic kidney disease",
        "Chronic obstructive pulmonary disease",
        "Complications of surgical/medical care",
        "Conduction disorders",
        "Congestive heart failure; nonhypertensive",
        "Coronary atherosclerosis and related",
        "Diabetes mellitus with complications",
        "Diabetes mellitus without complication",
        "Disorders of lipid metabolism",
        "Essential hypertension",
        "Fluid and electrolyte disorders",
        "Gastrointestinal hemorrhage",
        "Hypertension with complications",
        "Other liver diseases",
        "Other lower respiratory disease",
        "Other upper respiratory disease",
        "Pleurisy; pneumothorax; pulmonary collapse",
        "Pneumonia",
        "Respiratory failure; insufficiency; arrest",
        "Septicemia (except in labor)",
        "Shock"
    ]

    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, LORA_WEIGHTS_PATH, device)

    # 定义并加载预训练的分类头（只保存了 state_dict）
    num_classes = len(labels)
    classification_head = nn.Linear(model.config.hidden_size, num_classes)
    if not os.path.exists(CLASSIFICATION_HEAD_PATH):
        raise FileNotFoundError(
            f"分类头权重文件 {CLASSIFICATION_HEAD_PATH} 不存在，请先运行 train_teacher_head.py 进行训练。"
        )
    state_dict = torch.load(CLASSIFICATION_HEAD_PATH, map_location="cpu")
    classification_head.load_state_dict(state_dict)
    classification_head = classification_head.to(device)

    classification_model = LoRAClassificationModel(model, classification_head)
    classification_model.to(device).eval()

    # 分别为 train / val 生成软标签
    for split in ["train", "val"]:
        input_file_path = os.path.join(DATA_DIR, f"name_{split}.json")
        if not os.path.exists(input_file_path):
            print(f"文件 {input_file_path} 不存在，跳过该划分。")
            continue

        jsonl_data = read_jsonl(input_file_path)

        sample_label_probabilities = []
        for item in jsonl_data:
            batch_sample_probs = process_item(item, tokenizer, classification_model, device, labels)
            sample_label_probabilities.extend(batch_sample_probs.values())

        output_file_path = os.path.join(DATA_DIR, f"hands_name_{split}.json")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(sample_label_probabilities, f, ensure_ascii=False, indent=4)
        print(f"结果已保存到 {output_file_path}")


if __name__ == "__main__":
    main()



