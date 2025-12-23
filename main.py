import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import json
from sklearn.metrics import f1_score, accuracy_score, average_precision_score, roc_auc_score
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split

DATA_DIR = "./data"
TRAIN_FILE = os.path.join(DATA_DIR, "name_train.json")
VAL_FILE = os.path.join(DATA_DIR, "name_val.json")
TRAIN_SOFT_FILE = os.path.join(DATA_DIR, "hands_name_train.json")
VAL_SOFT_FILE = os.path.join(DATA_DIR, "hands_name_val.json")

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子
seed = 42
set_seed(seed)

# 定义标签与疾病的映射关系
disease_labels = [
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

def diseases_to_one_hot(disease_list, all_labels):
    return [1 if label in disease_list else 0 for label in all_labels]

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line:
                try:
                    data.append(json.loads(stripped_line))
                except json.JSONDecodeError as e:
                    print(f"Failed to parse line: {e}")
    return data

def read_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def clean_and_split(diseases_str, key_name):
    if not isinstance(diseases_str, str):
        print(f"Warning: Expected string for {key_name}, got {type(diseases_str)} instead.")
        return []

    prefix = "Disease that the patient may acquire: "
    if diseases_str.startswith(prefix):
        diseases_str = diseases_str[len(prefix):]

    if not diseases_str.strip():
        return []

    diseases_list = [d.strip() for d in diseases_str.split(",") if d.strip()]
    return diseases_list

def extract_disease_names_from_list(data_list):
    updated_data = []
    for item in data_list:
        updated_item = {"label": [], "input": []}
        if "output" in item:
            updated_item["label"] = clean_and_split(item["output"], "output")
        if "input" in item:
            updated_item["input"] = item["input"]
        updated_data.append(updated_item)
    return updated_data

class MyDataset(Dataset):
    def __init__(self, texts, labels_hard, labels_soft, tokenizer, max_length=512):
        self.texts = texts
        self.labels_hard = labels_hard
        self.labels_soft = labels_soft
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label_hard = self.labels_hard[idx]
        label_soft = self.labels_soft[idx]
        if isinstance(label_soft, dict):
            label_soft = list(label_soft.values())
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        label_hard = torch.tensor(label_hard, dtype=torch.float32)
        label_soft = torch.tensor(label_soft, dtype=torch.float32)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label_hard': label_hard,
            'label_soft': label_soft
        }

def custom_loss(logits, labels_hard, labels_soft, alpha=0.9):
    loss_fn_hard = nn.BCEWithLogitsLoss()
    loss_hard = loss_fn_hard(logits, labels_hard).float()

    loss_fn_soft = nn.BCELoss()
    probs = torch.sigmoid(logits)
    loss_soft = loss_fn_soft(probs, labels_soft).float()

    total_loss = alpha * loss_hard + (1 - alpha) * loss_soft
    return total_loss, loss_hard, loss_soft

def calculate(y_true, y_pred, args):
    y_true = np.array(y_true)
    pred = np.array(y_pred).astype(int)
    correct = (pred == y_true)

    total_acc = []
    total_f1 = []
    for i in range(args.num_labels):
        accuracy = (pred[:, i] == y_true[:, i]).mean()
        total_acc.append(accuracy)
        f1_macro = f1_score(y_true[:, i], pred[:, i], average='macro')
        total_f1.append(f1_macro)

    accuracy = correct.mean()
    f1_macro = f1_score(y_true, pred, average='macro')

    total_auc = []
    for i in range(args.num_labels):
        unique_labels = np.unique(y_true[:, i])
        if len(unique_labels) < 2:
            print(f"Warning: For label {i}, only one class present in y_true: {unique_labels}")
            total_auc.append(0.5)
        else:
            roc_auc = roc_auc_score(y_true[:, i], pred[:, i])
            total_auc.append(roc_auc)

    roc_auc = roc_auc_score(y_true.reshape(-1), pred.reshape(-1))

    total_aupr = []
    for i in range(args.num_labels):
        aupr = average_precision_score(y_true[:, i], pred[:, i])
        total_aupr.append(aupr)
    aupr = average_precision_score(y_true.reshape(-1), pred.reshape(-1))

    return accuracy, roc_auc, aupr, f1_macro

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=10, alpha=0.9, save_dir='./results/heads_results0.9'):
    best_f1 = 0  # 用于保存最佳验证集 F1 分数
    best_model_path = os.path.join('./results/heads_results0.9', 'best_model.pt')  # 最佳模型保存路径
    # 初始化 best_acc 变量
    best_acc = 0.0  # 假设我们想要最大化准确率

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_train_loss = 0
        total_train_loss_hard = 0
        total_train_loss_soft = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label_hard = batch['label_hard'].to(device)
            label_soft = batch['label_soft'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss, loss_hard, loss_soft = custom_loss(logits, label_hard, label_soft, alpha=alpha)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_loss_hard += loss_hard.item()
            total_train_loss_soft += loss_soft.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_loss_hard = total_train_loss_hard / len(train_loader)
        avg_train_loss_soft = total_train_loss_soft / len(train_loader)

        print(f"alpha：{alpha}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Training Hard Label Loss: {avg_train_loss_hard:.4f}")
        print(f"Training Soft Label Loss: {avg_train_loss_soft:.4f}")


        # 在每个 epoch 后验证验证集
        val_acc = validate_model(model, val_loader, device, alpha=alpha, save_dir=save_dir)

        # 如果当前验证集准确率更高，则保存模型
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with Validation Accuracy: {best_acc:.4f}")

    print(f"Training complete. Best Validation Accuracy: {best_acc:.4f}")

def validate_model(model, val_loader, device, alpha=0.9, save_dir='./results/heads_results0.9'):
    model.eval()
    total_val_loss = 0
    total_val_loss_hard = 0
    total_val_loss_soft = 0
    all_preds = []
    all_labels_hard = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label_hard = batch['label_hard'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss, loss_hard, loss_soft = custom_loss(logits, label_hard, label_hard, alpha=alpha)
            total_val_loss += loss.item()
            total_val_loss_hard += loss_hard.item()
            total_val_loss_soft += loss_soft.item()

            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)
            all_labels_hard.append(label_hard.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels_hard = np.concatenate(all_labels_hard, axis=0)

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_loss_hard = total_val_loss_hard / len(val_loader)
    avg_val_loss_soft = total_val_loss_soft / len(val_loader)

    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Hard Label Loss: {avg_val_loss_hard:.4f}")
    print(f"Validation Soft Label Loss: {avg_val_loss_soft:.4f}")

    binary_preds = (all_preds > 0.5).astype(int)
    args = type('Args', (object,), {'num_labels': 25})
    accuracy, roc_auc, aupr, f1_macro = calculate(all_labels_hard, binary_preds, args)

    print(f"Validation F1: {f1_macro:.4f}, Accuracy: {accuracy:.4f}, AUPR: {aupr:.4f}, AUC: {roc_auc:.4f}")

    return accuracy  # 返回验证集的准确率


def test_model(model, test_loader, device, alpha=0.9, save_dir='./results/heads_results0.9'):
    model.eval()
    total_test_loss = 0
    total_test_loss_hard = 0
    total_test_loss_soft = 0
    all_preds = []
    all_labels_hard = []
    all_labels_soft = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label_hard = batch['label_hard'].to(device)
            label_soft = batch['label_soft'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss, loss_hard, loss_soft = custom_loss(logits, label_hard, label_soft, alpha=alpha)
            total_test_loss += loss.item()
            total_test_loss_hard += loss_hard.item()
            total_test_loss_soft += loss_soft.item()

            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)
            all_labels_hard.append(label_hard.cpu().numpy())
            all_labels_soft.append(label_soft.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels_hard = np.concatenate(all_labels_hard, axis=0)
    all_labels_soft = np.concatenate(all_labels_soft, axis=0)

    intermediate_results = {
        'all_preds': all_preds,
        'all_labels_hard': all_labels_hard,
        'all_labels_soft': all_labels_soft
    }
    os.makedirs(save_dir, exist_ok=True)
    torch.save(intermediate_results, os.path.join(save_dir, f'test_results.pt'))

    avg_test_loss = total_test_loss / len(test_loader)
    avg_test_loss_hard = total_test_loss_hard / len(test_loader)
    avg_test_loss_soft = total_test_loss_soft / len(test_loader)

    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Hard Label Loss: {avg_test_loss_hard:.4f}")
    print(f"Test Soft Label Loss: {avg_test_loss_soft:.4f}")

    binary_preds = (all_preds > 0.5).astype(int)
    args = type('Args', (object,), {'num_labels': 25})
    accuracy, roc_auc, aupr, f1_macro = calculate(all_labels_hard, binary_preds, args)

    print(f"Test F1: {f1_macro:.4f}, Accuracy: {accuracy:.4f}, AUPR: {aupr:.4f}, AUC: {roc_auc:.4f}")
    # 定义保存路径
    results_txt_path = os.path.join('./results', 'test_metrics.txt')

    # 追加写入文本文件
    with open(results_txt_path, 'a') as f:
        f.write(f"alpha：{alpha}，Test F1: {f1_macro:.4f}, Accuracy: {accuracy:.4f}, AUPR: {aupr:.4f}, AUC: {roc_auc:.4f}\n")

    print(f"Test metrics appended to {results_txt_path}")


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载BERT模型和Tokenizer
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=25).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 读取训练和验证数据
    train_jsonl_data = read_jsonl(TRAIN_FILE)
    val_jsonl_data = read_jsonl(VAL_FILE)

    # 合并训练和验证数据
    combined_data = train_jsonl_data + val_jsonl_data

    # 读取训练和验证的软标签数据为列表
    train_soft_label_all = read_json_file(TRAIN_SOFT_FILE)
    val_soft_label_all = read_json_file(VAL_SOFT_FILE)

    # 合并软标签数据
    combined_soft_label = train_soft_label_all + val_soft_label_all

    # 提取疾病名称
    combined_extracted_data = extract_disease_names_from_list(combined_data)

    # 将硬标签数据转换为一个热编码格式
    combined_one_hot_encoded = [{
        "label": diseases_to_one_hot(item["label"], disease_labels),
        "input": item["input"]
    } for item in combined_extracted_data]

    # 使用索引划分数据集，确保软标签和硬标签对齐
    indices = np.arange(len(combined_one_hot_encoded))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=seed)

    def select_by_index(data_list, index_array):
        return [data_list[i] for i in index_array]

    train_data = select_by_index(combined_one_hot_encoded, train_idx)
    val_data = select_by_index(combined_one_hot_encoded, val_idx)
    test_data = select_by_index(combined_one_hot_encoded, test_idx)

    train_soft_label = select_by_index(combined_soft_label, train_idx)
    val_soft_label = select_by_index(combined_soft_label, val_idx)
    test_soft_label = select_by_index(combined_soft_label, test_idx)

    # 创建数据集和数据加载器
    train_dataset = MyDataset([item["input"] for item in train_data],
                              [item["label"] for item in train_data],
                              train_soft_label, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    val_dataset = MyDataset([item["input"] for item in val_data],
                            [item["label"] for item in val_data],
                            val_soft_label, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    test_dataset = MyDataset([item["input"] for item in test_data],
                             [item["label"] for item in test_data],
                             test_soft_label, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    # 训练模型
    train_model(model, train_loader, val_loader, optimizer, device, num_epochs=10, alpha=0.9, save_dir='./results/heads_results0.9')

    # 加载最佳模型
    best_model_path = os.path.join('./results/heads_results0.9', 'best_model.pt')
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)  # 确保模型在正确的设备上
    # 测试模型
    test_model(model, test_loader, device, alpha=0.9, save_dir='./results/heads_results0.9')


if __name__ == "__main__":
    main()

