# -*- coding: utf-8 -*-
# file: train_first_stage.py

import itertools
import logging
import argparse
import math
import os
import sys
import random
import numpy

from sklearn import metrics
from time import strftime, localtime

from transformers import BertModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import Tokenizer4Bert, ABSADataset7, ABSADataset6, ISADataset6, ISADataset1
from models import BERT_SPC, BERT_SPC1, BERT_SPCno, BERT_SPCno1

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model1 = opt.model_class1(bert, opt).to(opt.device)

        # outname = '{}-{}-{}-{}.txt'.format(opt.model_name, opt.dataset, opt.beta, strftime("%y%m%d-%H%M", localtime()))
        if opt.task == 'absa':
            self.trainset = ABSADataset6(opt.dataset_file['train'], self.opt.aug_multi, tokenizer)
            self.testset = ABSADataset7(opt.dataset_file['test'], tokenizer)
        elif opt.task == 'isa':
            self.trainset = ISADataset6(opt.dataset_file['train'], self.opt.aug_multi, tokenizer)
            self.testset = ISADataset1(opt.dataset_file['test'], tokenizer)

        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset) - valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()
        self.softmax_1 = nn.Softmax(dim=1)

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model1.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info(
            '> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model1.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def get_mean_wo_i(self, inputs, i):
        return (sum(inputs) - inputs[i]) / float(len(inputs) - 1)

    # stage 1
    def distloss1(self, xk, z_scores):
        gloss = []

        lonum = 1 + 4 * self.opt.aug_multi  # augnum + formernum
        for i in range(1, lonum):  # augnum + formernum 遍历augnum个计算差距
            reg_loss = torch.nn.functional.smooth_l1_loss(xk[i] * z_scores[0],
                                                          xk[0] * z_scores[i])  # xr <--> alphar * x
            gloss.append(reg_loss)
        return sum(gloss) / len(gloss)  # augnum+1求平均

    def regularization1(self, k, all_alphas, bertx):
        batch_size = k[0]
        totnum = k[1]
        bertx = bertx.reshape(batch_size, totnum, self.opt.bert_dim)

        # batch_size组数据计算组内差距
        all_regs1 = []
        for i in range(batch_size):
            regloss1 = self.distloss1(bertx[i], all_alphas)  # i, 1:
            all_regs1.append(regloss1)
        return sum(all_regs1) / len(all_regs1)  # batch_size求平均

    def _train1(self, train_data_loader):
        path = None
        global_step = 0
        self.model1.bert.requires_grad_(False)
        # self.model.dropout.requires_grad_(False)
        self.model1.dense.requires_grad_(False)
        _params1 = filter(lambda p: p.requires_grad, self.model1.parameters())
        optimizer1 = self.opt.optimizer1(_params1, lr=self.opt.lr, weight_decay=self.opt.l2reg)

        for i_epoch in range(self.opt.num_epoch_alpha):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model1.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                # optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                a = inputs[0].shape
                batch_size = inputs[0].shape[0]
                totnum = inputs[0].shape[1]
                inputs[0] = inputs[0].view(batch_size * totnum, self.opt.max_seq_len)
                if self.opt.task == 'absa':
                    inputs[1] = inputs[1].view(batch_size * totnum, self.opt.max_seq_len)
                outputs, alpha, bertx = self.model1(inputs)
                outputs_ans = outputs[0:a[0] * a[1]:(1 + 4 * self.opt.aug_multi)]

                alpha_loss = self.opt.beta1 * self.regularization1(a, alpha, bertx)
                loss = alpha_loss

                loss = loss / self.opt.accmulation_steps
                loss.backward()

                if ((i_batch + 1) % self.opt.accmulation_steps) == 0:
                    optimizer1.step()
                    optimizer1.zero_grad()

                n_total += len(outputs_ans)
                loss_total += loss.item() * len(outputs_ans)
                if global_step % self.opt.log_step == 0:
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}'.format(train_loss))
            logger.info('>> constant: {}'.format(self.model1.constant))

        alpha_trained = self.model1.constant
        alpha1 = alpha_trained[0]
        logger.info('>> saved: {}'.format(alpha_trained))
        print('alpha_tensor', id(alpha_trained))
        alpha_new = alpha_trained / alpha1
        logger.info('>> saved: {}'.format(alpha_new))
        alpha_file = './alpha/{}-new{}.pt'.format(self.opt.dataset, str(self.opt.num_epoch_alpha))
        torch.save(alpha_new, alpha_file)  # 保存
        print('alpha_list', id(alpha_new))
        return alpha_new

    def run(self):
        # Loss and Optimizer

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        # test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        # val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        self._train1(train_data_loader)


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name1', default='bert_spc1', type=str)
    parser.add_argument('--aug_multi', default='1', type=int, help='1, 2, 3, 4')
    parser.add_argument('--task', default='absa', type=str, help='isa, absa')
    parser.add_argument('--dataset', default='semrest_edarsrd1', type=str, help='twitter, restaurant, laptop, entity1')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch_alpha', default=10, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=8, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--accmulation_steps', default=2, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=85, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float,
                        help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--beta1', default=0.1, type=float)
    # parser.add_argument('--beta2', default=0.1, type=float)
    # parser.add_argument('--aug', default='all', type=str,
    #                    help='xxx_insert, bert_substitute, bert_insert, word2vec_insert')
    # parser.add_argument('--aug_ratio', default='0.1', type=str, help='0.1, 0.2, 0.3, 0.4')

    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        'bert_spc': BERT_SPC,
        'bert_spc1': BERT_SPC1,
        'bert_spcno': BERT_SPCno,
        'bert_spcno1': BERT_SPCno1,
    }
    dataset_files = {
        'restaurant': {
            'train': './datasets/semeval14/Restaurants_Train.xml.seg',
            'test': './datasets/semeval14/restsem_ise.txt'
        },
        'laptop': {
            'train': './datasets/semeval14/Laptops_Train.xml.seg',
            'test': './datasets/semeval14/laptopsem_ise.txt'
        },
        'semlaptop_1': {
            'train': './datasets/semeval14/semlaptop_1.txt',
            'test': './datasets/semeval14/laptopsem_ise.txt'
        },
        'semlaptop_2': {
            'train': './datasets/semeval14/semlaptop_2.txt',
            'test': './datasets/semeval14/laptopsem_ise.txt'
        },
        'semlaptop_4': {
            'train': './datasets/semeval14/semlaptop_4.txt',
            'test': './datasets/semeval14/laptopsem_ise.txt'
        },
        'semlaptop_8': {
            'train': './datasets/semeval14/semlaptop_8.txt',
            'test': './datasets/semeval14/laptopsem_ise.txt'
        },
        'semrest_1': {
            'train': './datasets/semeval14/semrest_1.txt',
            'test': './datasets/semeval14/restsem_ise.txt'
        },
        'semrest_2': {
            'train': './datasets/semeval14/semrest_2.txt',
            'test': './datasets/semeval14/restsem_ise.txt'
        },
        'semrest_4': {
            'train': './datasets/semeval14/semrest_4.txt',
            'test': './datasets/semeval14/restsem_ise.txt'
        },
        'semrest_8': {
            'train': './datasets/semeval14/semrest_8.txt',
            'test': './datasets/semeval14/restsem_ise.txt'
        },
        'entity_1': {
            'train': './datasets/semeval15/entity_1.txt',
            'test': './datasets/semeval15/entity_test1.txt'
        },
        'entity_2': {
            'train': './datasets/semeval15/entity_2.txt',
            'test': './datasets/semeval15/entity_test1.txt'
        },
        'entity_4': {
            'train': './datasets/semeval15/entity_4.txt',
            'test': './datasets/semeval15/entity_test1.txt'
        },

    }
    input_colses = {
        'bert_spc': ['concat_bert_indices', 'concat_segments_indices'],
        'bert_spc1': ['concat_bert_indices', 'concat_segments_indices'],
        'bert_spcno': ['concat_bert_indices'],
        'bert_spcno1': ['concat_bert_indices'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class1 = model_classes[opt.model_name1]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name1]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer1 = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = '{}-{}-{}.log'.format(opt.model_name1, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
