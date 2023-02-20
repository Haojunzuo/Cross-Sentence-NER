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
        return text_index.reshape(-1), label_index, len(label), index

    def __len__(self):
        return self.all_text.__len__()

class Bert_LSTM_NerModel(nn.Module):
    def __init__(self, lstm_hidden, class_num, train_wwm_unsupervised, train_glyce_unsupervised, dev_wwm_unsupervised,dev_glyce_unsupervised, windowSize):
        super().__init__()

        self.train_wwm_unsupervised = train_wwm_unsupervised
        self.train_glyce_unsupervised = train_glyce_unsupervised
        self.dev_wwm_unsupervised = dev_wwm_unsupervised
        self.dev_glyce_unsupervised = dev_glyce_unsupervised
        self.windowSize = windowSize
        self.bert = BertModel.from_pretrained('/home/cjh/NERCode/Chinese-BERT-base')
        for name, param in self.bert.named_parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(768, lstm_hidden, batch_first=True, num_layers=1,
                            bidirectional=True)  # 768 * lstm_hidden
        self.classifier = nn.Linear(256, class_num)
        self.crf = CRF(class_num, batch_first=True)
        self.linear = nn.Linear(1536, 768)
        # 将交叉熵损失替换为CRF来计算损失
        # self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, batch_index, sentence_index, batch_label=None, batch_type=None):
        bert_out = self.bert(batch_index)
        bert_out0, bert_out1 = bert_out[0], bert_out[1]  # bert_out0:字符级别特征, bert_out1:篇章级别
        cur_batch_size = bert_out0.shape[0]
        start = sentence_index[0] + 8
        end = sentence_index[-1] + 9
        tensorNext = torch.empty((0, 25), dtype=torch.float32).to(device)
        tensorTrain = torch.empty((0, 25), dtype=torch.float32).to(device)
        if batch_type == 'train':
            tensorLast = self.train_wwm_unsupervised[start - self.windowSize: end - self.windowSize].reshape(cur_batch_size, 1, 768)
            tensorLast = torch.cat([tensorLast, self.train_glyce_unsupervised[start - self.windowSize: end - self.windowSize].reshape(cur_batch_size, 1, 768)], dim=2)
            for i in reversed(range(self.windowSize-1)):
                tempLast1 = self.train_wwm_unsupervised[start - i-1: end - i-1].reshape(cur_batch_size, 1, 768)
                tempLast2 = self.train_glyce_unsupervised[start - i - 1: end - i - 1].reshape(cur_batch_size, 1, 768)
                tempLast3 = torch.cat([tempLast1, tempLast2], dim=2)
                tensorLast = torch.cat([tensorLast, tempLast3], dim=1)
            tensorLast = self.linear(tensorLast)
            tensorTrain = torch.cat([tensorLast, bert_out0], dim=1)
            for i in range(self.windowSize):
                tempNext1 = self.train_wwm_unsupervised[start + i+1: end + i+1].reshape(cur_batch_size, 1, 768)
                tempNext2 = self.train_glyce_unsupervised[start + i + 1: end + i + 1].reshape(cur_batch_size, 1, 768)
                tempNext3 = torch.cat([tempNext1, tempNext2], dim=2)
                if i == 0:
                    tensorNext = tempNext3
                else:
                    tensorNext = torch.cat([tensorNext, tempNext3], dim=1)
            tensorNext = self.linear(tensorNext)
            tensorTrain = torch.cat([tensorTrain, tensorNext], dim=1)
        if batch_type == 'dev':
            tensorLast = self.dev_wwm_unsupervised[start - self.windowSize: end - self.windowSize].reshape( cur_batch_size, 1, 768)
            tensorLast = torch.cat([tensorLast, self.dev_glyce_unsupervised[ start - self.windowSize: end - self.windowSize].reshape(cur_batch_size, 1, 768)], dim=2)
            for i in reversed(range(self.windowSize - 1)):
                tempLast1 = self.dev_wwm_unsupervised[start - i - 1: end - i - 1].reshape(cur_batch_size, 1, 768)
                tempLast2 = self.dev_glyce_unsupervised[start - i - 1: end - i - 1].reshape(cur_batch_size, 1, 768)
                tempLast3 = torch.cat([tempLast1, tempLast2], dim=2)
                tensorLast = torch.cat([tensorLast, tempLast3], dim=1)
            tensorLast = self.linear(tensorLast)
            tensorTrain = torch.cat([tensorLast, bert_out0], dim=1)
            for i in range(self.windowSize):
                tempNext1 = self.dev_wwm_unsupervised[start + i + 1: end + i + 1].reshape(cur_batch_size, 1, 768)
                tempNext2 = self.dev_glyce_unsupervised[start + i + 1: end + i + 1].reshape(cur_batch_size, 1, 768)
                tempNext3 = torch.cat([tempNext1, tempNext2], dim=2)
                if i == 0:
                    tensorNext = tempNext3
                else:
                    tensorNext = torch.cat([tensorNext, tempNext3], dim=1)
            tensorNext = self.linear(tensorNext)
            tensorTrain = torch.cat([tensorTrain, tensorNext], dim=1)
        lstm_out, _ = self.lstm(tensorTrain)
        indices = torch.tensor([i for i in range(self.windowSize * 2, 32+self.windowSize * 2)]).to(device)
        lstm_out_reshape = torch.index_select(lstm_out, 1, indices)
        pre = self.classifier(lstm_out_reshape)
        if batch_label is not None:
            loss = -self.crf(pre, batch_label)
            return loss
        else:
            # 相较于不使用CRF，需要加上decode
            pre = self.crf.decode(pre)
            # return torch.argmax(pre, dim=-1)
            # 直接输出
            return pre

if __name__ == "__main__":

    batch_size = 50
    epoch = 100
    max_len = 30
    lr = 0.0005
    lstm_hidden = 128
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    windowSize = [1,2,3,4]
    scoreListAll = []
    for w in windowSize:
        train_wwm_unsupervised = torch.load('../tensor/sentenceRep/train-bert-wwm-unsupervised.pt')
        tailTrain = train_wwm_unsupervised[-8:]
        headTrain = train_wwm_unsupervised[0:8]
        tempTensor = torch.cat([tailTrain, train_wwm_unsupervised], dim=0)
        train_wwm_unsupervised = torch.cat([tempTensor, headTrain], dim=0).to(device)

        train_glyce_unsupervised = torch.load('../tensor/sentenceRep/train-glyce-unsupervised.pt')
        tailTrain = train_glyce_unsupervised[-8:]
        headTrain = train_glyce_unsupervised[0:8]
        tempTensor = torch.cat([tailTrain, train_glyce_unsupervised], dim=0)
        train_glyce_unsupervised = torch.cat([tempTensor, headTrain], dim=0).to(device)

        dev_wwm_unsupervised = torch.load('../tensor/sentenceRep/dev-bert-wwm-unsupervised.pt')
        tailDev = dev_wwm_unsupervised[-8:]
        headDev = dev_wwm_unsupervised[0:8]
        tempTensor = torch.cat([tailDev, dev_wwm_unsupervised], dim=0)
        dev_wwm_unsupervised = torch.cat([tempTensor, headDev], dim=0).to(device)

        dev_glyce_unsupervised = torch.load('../tensor/sentenceRep/dev-glyce-unsupervised.pt')
        tailTrain = dev_glyce_unsupervised[-8:]
        headTrain = dev_glyce_unsupervised[0:8]
        tempTensor = torch.cat([tailTrain, dev_glyce_unsupervised], dim=0)
        dev_glyce_unsupervised = torch.cat([tempTensor, headTrain], dim=0).to(device)

        train_text, train_label = read_data(os.path.join("../data", "train.txt"))
        dev_text, dev_label = read_data(os.path.join("../data", "dev.txt"))
        test_text, test_label = read_data(os.path.join("../data", "test.txt"))
        label_2_index, index_2_label = build_label(train_label)
        tokenizer = BertTokenizer.from_pretrained('/home/cjh/NERCode/Chinese-BERT-base')


        train_dataset = BertDataset(train_text, train_label, label_2_index, max_len, tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        dev_dataset = BertDataset(dev_text, dev_label, label_2_index, max_len, tokenizer)
        dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

        model = Bert_LSTM_NerModel(lstm_hidden, len(label_2_index), train_wwm_unsupervised, train_glyce_unsupervised, dev_wwm_unsupervised,dev_glyce_unsupervised, w).to(device)
        opt = AdamW(model.parameters(), lr)

        best_f1_score = 0
        scoreList = []
        for e in range(epoch):
            model.train()
            for batch_text_index, batch_label_index, batch_len, sentence_index in train_dataloader:
                batch_text_index = batch_text_index.to(device)
                batch_label_index = batch_label_index.to(device)
                loss = model.forward(batch_index=batch_text_index, batch_label=batch_label_index, sentence_index=sentence_index, batch_type='train')
                loss.backward()

                opt.step()
                opt.zero_grad()

                print(f'loss:{loss:.2f}')

            model.eval()

            all_pre = []
            all_tag = []
            for batch_text_index, batch_label_index, batch_len, sentence_index in dev_dataloader:
                batch_text_index = batch_text_index.to(device)
                batch_label_index = batch_label_index.to(device)
                pre = model.forward(batch_text_index, sentence_index=sentence_index, batch_type='dev')

                # 使用CRF之后，此处不需再转为list
                # pre = pre.cpu().numpy().tolist()
                tag = batch_label_index.cpu().numpy().tolist()

                # 消除pad和特殊字符影响，pad和特殊字符不参与F1分数计算
                for p, t, l in zip(pre, tag, batch_len):
                    p = p[1:1 + l]
                    t = t[1:1 + l]

                    p = [index_2_label[i] for i in p]
                    t = [index_2_label[i] for i in t]

                    all_pre.append(p)
                    all_tag.append(t)

            f1_score = seq_f1_score(all_tag, all_pre)
            if f1_score > best_f1_score:
                best_f1_score = f1_score
            scoreList.append(f1_score)
            print("f1:{}, best_f1:{}".format(f1_score, best_f1_score))
        scoreList.sort(reverse=True)
        scoreListAll.append(scoreList)
        print(scoreList)
    print(scoreListAll)