import os
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt import PromptDataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 42
batch_size = 16
num_class = 2
max_seq_l = 512
lr = 2e-5
num_epochs = 10
use_cuda = True
model_name = "roberta"
pretrainedmodel_path = "./pretrained-roberta-large"  # the path of the pre-trained model
dataset_path = "../data/temp_data.csv"

# 读取 Excel 文件
df = pd.read_csv(dataset_path)

t_dataset = df.loc[df['target'] == 1]
f_dataset = df.loc[df['target'] == 0]
datasets = f_dataset.append(t_dataset)
datasets = datasets[["modified_column","target"]]
dataset = Dataset.from_dict(datasets)

traintest = dataset.train_test_split(test_size=0.2, seed=seed)
validationtest = traintest['test'].train_test_split(test_size=0.5, seed=seed)
train_val_test = {}
train_val_test['train'] = traintest['train']
train_val_test['validation'] = validationtest['train']
train_val_test['test'] = validationtest['test']

dataset = {}
for split in ['train', 'validation', 'test']:
    dataset[split] = []
    for data in train_val_test[split]:
        input_example = InputExample(text_a = data['modified_column'], label=int(data['target']))
        dataset[split].append(input_example)

# load plm
plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "../CodeBERT")

# construct template
template_text = 'Here is {"mask"} vulnerability. {"placeholder":"text_a"}'
mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)

# define the verbalizer
from openprompt.prompts import ManualVerbalizer
myverbalizer = ManualVerbalizer(tokenizer, num_classes=num_class, label_words=[["true","yes"], ["false","no"]])

# define prompt model for classification
from openprompt import PromptForClassification
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
checkpoint = torch.load('../prompt/best.ckpt')
prompt_model.load_state_dict(checkpoint)
if use_cuda:
    prompt_model=  prompt_model.cuda()

# DataLoader
validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, batch_size=batch_size,shuffle=True,
    teacher_forcing=False, predict_eos_token=False, truncate_method="head")
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, batch_size=batch_size,shuffle=True,
    teacher_forcing=False, predict_eos_token=False, truncate_method="head")


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

result_Recall = 0
result_F1 = 0
result_Pre = 0
result_Acc = 0


# validate
print('\n\nepoch{}------------validate------------')
acc, recall, precision, f1 = test(prompt_model, validation_dataloader)
if result_F1 < f1:
    result_Recall = recall
    result_F1 = f1
    result_Pre = precision
    result_Acc = acc

# test
print('\n\nepoch{}------------test------------')
acc, recall, precision, f1 = test(prompt_model, test_dataloader)
if result_F1 < f1:
    result_Recall = recall
    result_F1 = f1
    result_Pre = precision
    result_Acc = acc
print("\n\n result acc:{}   recall:{}   precision:{}   f1:{}".format(result_Acc, result_Recall, result_Pre, result_F1))