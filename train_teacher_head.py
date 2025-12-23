import os
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# 与 hands.py / alpha1.0.9.py 保持一致的 25 个疾病标签
DISEASE_LABELS = [
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
    "Shock",
]


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LORA_WEIGHTS_PATH = "./lora/model"
DATA_DIR = "./data"
TRAIN_FILE = os.path.join(DATA_DIR, "name_train.json")
VAL_FILE = os.path.join(DATA_DIR, "name_val.json")
CLASSIFICATION_HEAD_PATH = "./lora/teacher_classification_head.pt"

MAX_LENGTH = 1024
BATCH_SIZE = 2  # 大模型，batch 适当小一点
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4


def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Failed to parse line {i} in {file_path}: {e}")
    return data


def clean_and_split(diseases_str, key_name):
    if not isinstance(diseases_str, str):
        print(f"Warning: Expected string for {key_name}, got {type(diseases_str)} instead.")
        return []

    prefix = "Disease that the patient may acquire: "
    if diseases_str.startswith(prefix):
        diseases_str = diseases_str[len(prefix) :]

    if not diseases_str.strip():
        return []

    return [d.strip() for d in diseases_str.split(",") if d.strip()]


def diseases_to_one_hot(disease_list, all_labels):
    return [1 if label in disease_list else 0 for label in all_labels]


class InstructionDataset(Dataset):
    def __init__(self, items, labels, tokenizer, max_length=MAX_LENGTH):
        self.items = items
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        instruction_part = item.get("instruction", "")
        input_part = item.get("input", "")
        instructions = f"{instruction_part}\n{input_part}" if input_part else instruction_part

        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": instructions}],
            tokenize=False,
            add_generation_prompt=True,
        )

        encoding = self.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }


class LoRAClassificationModel(nn.Module):
    def __init__(self, base_model, classification_head):
        super().__init__()
        self.base_model = base_model
        self.classification_head = classification_head

    def forward(self, input_ids, **kwargs):
        outputs = self.base_model(input_ids=input_ids, output_hidden_states=True, **kwargs)
        last_token_hidden_state = outputs.hidden_states[-1][:, -1, :]
        last_token_hidden_state = last_token_hidden_state.to(self.classification_head.weight.dtype)
        logits = self.classification_head(last_token_hidden_state)
        return logits


def load_model_and_tokenizer(model_name, lora_weights_path, device):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, lora_weights_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.to(device).eval()
    return model, tokenizer


def build_datasets(tokenizer):
    train_items = read_jsonl(TRAIN_FILE)
    val_items = read_jsonl(VAL_FILE)

    def build_labels(items):
        labels = []
        for item in items:
            diseases = clean_and_split(item.get("output", ""), "output")
            labels.append(diseases_to_one_hot(diseases, DISEASE_LABELS))
        return labels

    train_labels = build_labels(train_items)
    val_labels = build_labels(val_items)

    train_dataset = InstructionDataset(train_items, train_labels, tokenizer)
    val_dataset = InstructionDataset(val_items, val_labels, tokenizer)
    return train_dataset, val_dataset


def train_teacher_head():
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"Train file not found: {TRAIN_FILE}")
    if not os.path.exists(VAL_FILE):
        raise FileNotFoundError(f"Validation file not found: {VAL_FILE}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, LORA_WEIGHTS_PATH, device)

    num_classes = len(DISEASE_LABELS)
    classification_head = nn.Linear(model.config.hidden_size, num_classes)

    for param in model.parameters():
        param.requires_grad = False

    classification_head = classification_head.to(device)
    classification_model = LoRAClassificationModel(model, classification_head).to(device)

    train_dataset, val_dataset = build_datasets(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(classification_head.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(NUM_EPOCHS):
        classification_model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            logits = classification_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits.float(), labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(len(train_loader), 1)

        classification_model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = classification_model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits.float(), labels)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / max(len(val_loader), 1)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    save_dir = os.path.dirname(CLASSIFICATION_HEAD_PATH) or "."
    os.makedirs(save_dir, exist_ok=True)
    torch.save(classification_head.state_dict(), CLASSIFICATION_HEAD_PATH)
    print(f"Classification head saved to {CLASSIFICATION_HEAD_PATH}")


if __name__ == "__main__":
    train_teacher_head()
