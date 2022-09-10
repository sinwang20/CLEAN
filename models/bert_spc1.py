# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn


class BERT_SPC1(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC1, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.softmax = nn.Softmax(dim=0)
        self.constant = nn.Parameter(torch.tensor([1.0 for _ in range(1+4*opt.aug_multi)]))
        #self.constant = nn.Parameter(torch.tensor([1.0, 0.9, 0.9, 0.9, 0.9]))

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        _, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        #alpha = self.softmax(self.constant)
        return logits, self.constant, pooled_output
