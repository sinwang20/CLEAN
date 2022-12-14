# -*- coding: utf-8 -*-
# file: BERT_SPCno.py

import torch
import torch.nn as nn


class BERT_SPCno(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPCno, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices = inputs[0]
        _, pooled_output = self.bert(text_bert_indices, attention_mask=None)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits

