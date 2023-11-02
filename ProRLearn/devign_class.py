import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from torch.utils.data import DataLoader


device = torch.device("cuda")


model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


model = model.to(device)


datasets = pd.read_csv('../data/temp_data.csv')
train_data = datasets.sample(frac=0.8)
test_data = datasets.drop(train_data.index)
# train_data = train_data.sample(frac=0.75)
val_data = test_data.sample(frac=0.5)
test_data = test_data.drop(val_data.index)


def preprocess_data(texts, labels):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])
    labels = labels.to_list()
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    return input_ids, attention_masks, labels


train_input_ids, train_attention_masks, train_labels = preprocess_data(train_data['modified_column'], train_data['target'])
val_input_ids, val_attention_masks, val_labels = preprocess_data(val_data['modified_column'], val_data['target'])
test_input_ids, test_attention_masks, test_labels = preprocess_data(test_data['modified_column'], test_data['target'])

batch_size = 16
epochs = 4
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)


train_data = torch.utils.data.TensorDataset(train_input_ids, train_attention_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
val_data = torch.utils.data.TensorDataset(val_input_ids, val_attention_masks, val_labels)
val_dataloader = DataLoader(val_data, batch_size=16, shuffle=False)
test_data = torch.utils.data.TensorDataset(test_input_ids, test_attention_masks, test_labels)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=False)


total_steps = epochs * len(train_dataloader)
progress_bar = tqdm(range(total_steps))
best_f1 = 0
best_acc = 0
best_pre = 0
best_recall = 0
result_Recall = 0
result_F1 = 0
result_Pre = 0
result_Acc = 0

for epoch in range(epochs):

    model.train()
    total_loss = 0
    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_masks, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        progress_bar.update(1)
    avg_train_loss = total_loss / len(train_dataloader)


    model.eval()
    with torch.no_grad():
        val_predictions = []
        val_label = []
        for batch in val_dataloader:
            batch = tuple(t.to(device) for t in batch)
            val_input_ids, val_attention_masks, val_labels = batch
            outputs = model(val_input_ids, attention_mask=val_attention_masks, labels=val_labels)
            val_logits = outputs.logits
            predictions = torch.argmax(val_logits, dim=1).cpu().numpy()
            for i in predictions:
                val_predictions.append(i)
            val_labels = val_labels.cpu().numpy()
            for i in val_labels:
                val_label.append(i)
        val_f1 = f1_score(val_label, val_predictions)
        val_accuracy = accuracy_score(val_label, val_predictions)
        val_precision = precision_score(val_label, val_predictions)
        val_recall = recall_score(val_label, val_predictions)
        if(best_acc < val_accuracy):
            best_acc = val_accuracy
        if best_f1 < val_f1:
            best_f1 = val_f1
        if best_pre < val_precision:
            best_pre = val_precision
        if best_recall < val_recall:
            best_recall = val_recall
        if result_F1 < val_f1:
            result_Recall = val_recall
            result_F1 = val_f1
            result_Pre = val_precision
            result_Acc = val_accuracy

    model.eval()
    with torch.no_grad():
        test_predictions = []
        test_label = []
        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            test_input_ids, test_attention_masks, test_labels = batch
            outputs = model(test_input_ids, attention_mask=test_attention_masks, labels=test_labels)
            test_logits = outputs.logits
            predictions = torch.argmax(test_logits, dim=1).cpu().numpy()
            for i in predictions:
                test_predictions.append(i)
            test_labels = test_labels.cpu().numpy()
            for i in test_labels:
                test_label.append(i)
        test_f1 = f1_score(test_label, test_predictions)
        test_accuracy = accuracy_score(test_label, test_predictions)
        test_recall = recall_score(test_label, test_predictions)
        test_precision = precision_score(test_label, test_predictions)
        if (best_acc < val_accuracy):
            best_acc = val_accuracy
        if best_f1 < val_f1:
            best_f1 = val_f1
        if best_pre < val_precision:
            best_pre = val_precision
        if best_recall < val_recall:
            best_recall = val_recall
        if result_F1 < val_f1:
            result_Recall = val_recall
            result_F1 = val_f1
            result_Pre = val_precision
            result_Acc = val_accuracy
    print(f"Epoch {epoch + 1}:")
    print("\n\n best acc:{}   recall:{}   precision:{}   f1:{}".format(result_Acc, result_Recall, result_Pre, result_F1))
    print()

print("\n\n result acc:{}   recall:{}   precision:{}   f1:{}".format(result_Acc, result_Recall, result_Pre, result_F1))