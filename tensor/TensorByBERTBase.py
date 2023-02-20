import os

import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers import BertModel
from transformers import AdamW
import torch
import torch.nn as nn
from seqeval.metrics import accuracy_score as seq_accuracy_score
from seqeval.metrics import f1_score as seq_f1_score
from seqeval.metrics import precision_score as seq_precision_score
from seqeval.metrics import recall_score as seq_recall_score
import sys
from torchcrf import CRF

def read_data(file):
    with open(file, "r", encoding="utf-8") as f:
        all_data = f.read().split("\n")

    all_text = []
    all_label = []

    text = []
    label = []
    for data in all_data:
        if data == "":
            all_text.append(text)
            all_label.append(label)
            text = []
            label = []
        else:
            t, l = data.split(" ")
            text.append(t)
            label.append(l)

    return all_text, all_label

def build_label(train_label):
    label_2_index = {"PAD": 0, "UNK": 1}
    for label in train_label:
        for l in label:
            if l not in label_2_index:
                label_2_index[l] = len(label_2_index)
    return label_2_index, list(label_2_index)

class BertDataset(Dataset):
    def __init__(self, all_text, all_label, label_2_index, max_len, tokenizer):
        self.all_text = all_text
        self.all_label = all_label
        self.label_2_index = label_2_index
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        text = self.all_text[index]
        label = self.all_label[index][:self.max_len]

        text_index = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_len + 2,
                                           padding="max_length", truncation=True, return_tensors="pt")
        label_index = [0] + [self.label_2_index.get(l, 1) for l in label] + [0] + [0] * (self.max_len - len(text))

        label_index = torch.tensor(label_index)
        return text_index.reshape(-1), label_index, len(label)

    def __len__(self):
        return self.all_text.__len__()

class Bert_LSTM_NerModel(nn.Module):
    def __init__(self, lstm_hidden, class_num):
        super().__init__()

        self.bert = BertModel.from_pretrained('F:\\NER Embedding\\Chinese-BERT-base')
        for name, param in self.bert.named_parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(768, lstm_hidden, batch_first=True, num_layers=1,
                            bidirectional=True)  # 768 * lstm_hidden
        self.classifier = nn.Linear(256, class_num)
        self.crf = CRF(class_num, batch_first=True)
        self.relayTensor = torch.empty(0, 768).to(device)
        # 将交叉熵损失替换为CRF来计算损失
        # self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, batch_index, batch_label=None, saveTensor=False):
        bert_out = self.bert(batch_index)
        bert_out0, bert_out1 = bert_out[0], bert_out[1]  # bert_out0:字符级别特征, bert_out1:篇章级别
        shape = bert_out0.shape[0]
        if saveTensor:
            self.relayTensor = torch.cat([self.relayTensor, bert_out1], 0)
            if shape != 50:
                torch.save(self.relayTensor, './sentenceRep/dev-bert-base-supervised.pt')
        lstm_out, _ = self.lstm(bert_out0)
        pre = self.classifier(lstm_out)

        if batch_label is not None:
            # loss = self.loss_fun(pre.reshape(-1, pre.shape[-1]), batch_label.reshape(-1))
            loss = -self.crf(pre, batch_label)
            return loss
        else:
            # 相较于不使用CRF，需要加上decode
            pre = self.crf.decode(pre)
            # return torch.argmax(pre, dim=-1)
            # 直接输出
            return pre

if __name__ == "__main__":

    train_text, train_label = read_data(os.path.join("../data", "train.txt"))
    dev_text, dev_label = read_data(os.path.join("../data", "dev.txt"))
    test_text, test_label = read_data(os.path.join("../data", "test.txt"))
    label_2_index, index_2_label = build_label(train_label)
    tokenizer = BertTokenizer.from_pretrained('F:\\NER Embedding\\Chinese-BERT-base')

    batch_size = 50
    epoch = 50
    max_len = 30
    lr = 0.0005
    lstm_hidden = 128
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_dataset = BertDataset(train_text, train_label, label_2_index, max_len, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    dev_dataset = BertDataset(dev_text, dev_label, label_2_index, max_len, tokenizer)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    model = Bert_LSTM_NerModel(lstm_hidden, len(label_2_index)).to(device)
    opt = AdamW(model.parameters(), lr)

    for e in range(epoch):
        model.train()
        for batch_text_index, batch_label_index, batch_len in dev_dataloader:
            batch_text_index = batch_text_index.to(device)
            batch_label_index = batch_label_index.to(device)
            # loss = model.forward(batch_index=batch_text_index, batch_label=batch_label_index, saveTensor=True)
            if e == 49:
                loss = model.forward(batch_index=batch_text_index, batch_label=batch_label_index, saveTensor=True)
            else:
                loss = model.forward(batch_index=batch_text_index, batch_label=batch_label_index)
            loss.backward()

            opt.step()
            opt.zero_grad()

            print(f'loss:{loss:.2f}')

        # model.eval()
        #
        # all_pre = []
        # all_tag = []
        # for batch_text_index, batch_label_index, batch_len in dev_dataloader:
        #     batch_text_index = batch_text_index.to(device)
        #     batch_label_index = batch_label_index.to(device)
        #     pre = model.forward(batch_text_index)
        #
        #     # 使用CRF之后，此处不需再转为list
        #     # pre = pre.cpu().numpy().tolist()
        #     tag = batch_label_index.cpu().numpy().tolist()
        #
        #     # 消除pad和特殊字符影响，pad和特殊字符不参与F1分数计算
        #     for p, t, l in zip(pre, tag, batch_len):
        #         p = p[1:1 + l]
        #         t = t[1:1 + l]
        #
        #         p = [index_2_label[i] for i in p]
        #         t = [index_2_label[i] for i in t]
        #
        #         all_pre.append(p)
        #         all_tag.append(t)
        #
        # f1_score = seq_f1_score(all_tag, all_pre)
        # print(f"f1:{f1_score}")
