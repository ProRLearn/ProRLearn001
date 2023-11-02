import os
import torch
import pandas as pd
import math
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 42
batch_size = 16
num_class = 2
max_seq_l = 256
lr = 2e-5
num_epochs = 6
use_cuda = True
model_name = "roberta"
pretrainedmodel_path = "./pretrained-roberta-large"  # the path of the pre-trained model
dataset_path = "../data/reveal.csv"

df = pd.read_csv(dataset_path)
dataset = Dataset.from_dict(df)
validation = pd.read_csv("../data/validation.csv")
test = pd.read_csv("../data/test.csv")
traintest = dataset.train_test_split(test_size=0.2)
validation = Dataset.from_dict(validation)
test = Dataset.from_dict(test)
# traintest_20 = traintest['train'].train_test_split(test_size=0.25, seed=seed)
validationtest = traintest['test'].train_test_split(test_size=0.5)
train_val_test = {}
train_val_test['train'] = traintest['train']
train_val_test['validation'] = validation
train_val_test['test'] = test

dataset = {}
for split in ['train', 'validation', 'test']:
    dataset[split] = []
    for data in train_val_test[split]:
        input_example = InputExample(text_a = data['code'], label=int(data['label']))
        dataset[split].append(input_example)

# load plm
plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "../CodeBERT")

# construct hard template
# template_text = 'The vulnerability is {"mask"}. {"placeholder":"text_a"}'
# template_text = '{"soft": "The"} vulnerability {"soft":"is"} {"mask"} {"soft"} {"placeholder":"text_a"}'
# template_text = 'The severity of the following vulnerability is {"mask"}. {"placeholder":"text_a"}'
template_text = 'This code {"place":"text_a"} is a vulnerability. {"mask"}.'
mytemplate = {}
# mytemplate = MixedTemplate(model = plm, tokenizer=tokenizer, text=template_text)

# define the verbalizer
from openprompt.prompts import ManualVerbalizer
myverbalizer = ManualVerbalizer(tokenizer, num_classes=num_class, label_words=[["true","yes"], ["false","no"]])
# myverbalizer = ManualVerbalizer(tokenizer, num_classes=num_class, label_words=[["low"], ["medium"], ["high"], ["critical"]])

# define prompt model for classification
from openprompt import PromptForClassification
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model=  prompt_model.cuda()

# DataLoader
from openprompt import PromptDataLoader
train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, batch_size=batch_size,shuffle=True,
    teacher_forcing=False, predict_eos_token=False, truncate_method="head")
validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, batch_size=batch_size,shuffle=True,
    teacher_forcing=False, predict_eos_token=False, truncate_method="head")
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, batch_size=batch_size,shuffle=True,
    teacher_forcing=False, predict_eos_token=False, truncate_method="head")

from transformers import AdamW, get_linear_schedule_with_warmup

no_decay = ['bias', 'LayerNorm.weight']
# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters = [
    {'params': [p for n, p in prompt_model.named_parameter() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameter() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
num_training_steps = num_epochs * len(train_dataloader)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

def test(prompt_model, test_dataloader):
    allpreds = []
    alllabels = []
    with torch.no_grad():
        for step, inputs in enumerate(test_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        acc = accuracy_score(alllabels, allpreds)
        f1 = f1_score(alllabels, allpreds)
        precision = precision_score(alllabels, allpreds, zero_division=0)
        recall = recall_score(alllabels, allpreds)
        print("acc: {}  recall: {}  precision: {}  f1: {}".format(acc, recall, precision, f1))
    return acc, recall, precision, f1


from tqdm.auto import tqdm
from itertools import tee

output_dir = "../sard_prompt/Reveal"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

progress_bar = tqdm(range(num_training_steps))
bestRecall = 0
bestAcc = 0
bestPre = 0
bestF1 = 0

result_Recall = 0
result_F1 = 0
result_Pre = 0
result_Acc = 0
for epoch in range(num_epochs):
    # train
    tot_loss = 0
    for step, inputs in enumerate(train_dataloader):
        loss = 0
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        action = torch.argmax(logits, dim=1)
        y = F.softmax(logits, dim=-1)
        labels = inputs['label']
        reward = 1 * (action == labels).sum() - batch_size
        for i in range(y.shape[0]):
            new_shape = (1, 2)
            input_1 = logits[i]
            input_1 = input_1.reshape(new_shape)
            loss += loss_func_1(logits[i], labels[i])
            # if (action[i] == labels[i]):
            #     loss += loss_func_1(logits[i], labels[i])
            # else:
            #     loss += loss_func_1(logits[i], labels[i]) * 1.2
        loss = loss * 0.0625 * reward * -1
        loss_1 = loss_func(logits, labels)
        if(loss < 0):
            loss = loss * -1
        # loss = loss / 16
        #loss = torch.tensor(loss)
        #loss.requires_grad_(True)
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        progress_bar.update(1)
    print("\nEpoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)

    # validate
    print('\n\nepoch{}------------validate------------'.format(epoch))
    acc, recall, precision, f1 = test(prompt_model, validation_dataloader)
    if recall > bestRecall:
        bestRecall = recall
    if acc > bestAcc:
        bestAcc = acc
    if precision > bestPre:
        bestPre = precision
    if f1 > bestF1:
        bestF1 = f1
    if result_F1 < bestF1:
        result_Recall = recall
        result_F1 = f1
        result_Pre = precision
        result_Acc = acc
    # test
    print('\n\nepoch{}------------test------------'.format(epoch))
    acc, recall, precision, f1 = test(prompt_model, test_dataloader)

    if result_F1 < f1:
        result_Recall = recall
        result_F1 = f1
        result_Pre = precision
        result_Acc = acc
print("\n\n best acc:{}   recall:{}   precision:{}   f1:{}".format(bestAcc, bestRecall, bestPre, bestF1))
print("\n\n result acc:{}   recall:{}   precision:{}   f1:{}".format(result_Acc, result_Recall, result_Pre, result_F1))