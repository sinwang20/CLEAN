# -*- coding: utf-8 -*-
# file: train_second_stage.py

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
        self.model2 = opt.model_class2(bert, opt).to(opt.device)

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
        for p in self.model2.parameters():
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
        for child in self.model2.children():
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

    # stage 2
    def distloss2(self, xk, z_scores):
        gloss = []
        for i in range((1 + 4 * self.opt.aug_multi)):  # augnum+formernum 遍历augnum个计算差距
            reg_loss = torch.nn.functional.smooth_l1_loss(xk[i] * self.get_mean_wo_i(z_scores, i)
                                                          , self.get_mean_wo_i(xk, i) * z_scores[i])
            gloss.append(reg_loss)
        return sum(gloss) / len(gloss)  # augnum+1求平均

    def regularization2(self, k, all_outs, all_alphas):
        batch_size = k[0]
        totnum = k[1]
        all_outsi = self.softmax_1(all_outs)
        all_outsi = all_outsi.reshape(k[0], k[1], self.opt.polarities_dim)

        # batch_size组数据计算组内差距
        all_regs2 = []
        for i in range(batch_size):
            regloss2 = self.distloss2(all_outsi[i], all_alphas)  # i, 1:
            all_regs2.append(regloss2)
        return sum(all_regs2) / len(all_regs2)  # batch_size求平均

    def _train2(self, criterion, optimizer2, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        path = None
        global_step = 0
        alpha_file = './alpha/{}-{}.pt'.format(self.opt.dataset, str(self.opt.num_epoch_alpha))
        alpha = torch.load(alpha_file)  # 读取
        logger.info('>> alpha_read: {}'.format(alpha))
        alpha = alpha.tolist()
        print(alpha)
        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model2.train()

            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                a = inputs[0].shape
                batch_size = inputs[0].shape[0]
                totnum = inputs[0].shape[1]
                inputs[0] = inputs[0].view(batch_size * totnum, self.opt.max_seq_len)
                if self.opt.task == 'absa':
                    inputs[1] = inputs[1].view(batch_size * totnum, self.opt.max_seq_len)
                outputs = self.model2(inputs)
                outputs_ans = outputs[0:a[0] * a[1]:(1 + 4 * self.opt.aug_multi)]  # batch*polar
                targets = batch['polarity'].to(self.opt.device)

                loss = criterion(outputs_ans, targets) + self.opt.beta2 * self.regularization2(a, outputs, alpha)

                loss = loss / self.opt.accmulation_steps
                loss.backward()

                if ((i_batch + 1) % self.opt.accmulation_steps) == 0:
                    optimizer2.step()
                    optimizer2.zero_grad()

                n_correct += (torch.argmax(outputs_ans, -1) == targets).sum().item()
                n_total += len(outputs_ans)
                loss_total += loss.item() * len(outputs_ans)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            if self.opt.task == 'absa':
                val_acc, val_f1, ese, ise = self._evaluate_acc_f1(val_data_loader)
                logger.info(
                    '> val_acc: {:.4f}, val_f1: {:.4f}, ese: {:.4f}, ise: {:.4f}'.format(val_acc, val_f1, ese, ise))
            else:
                val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
                logger.info(
                    '> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = i_epoch
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_val_acc_{2}_{3}'.format(self.opt.model_name2, self.opt.dataset,
                                                                   round(val_acc, 4), round(val_f1, 4))
                torch.save(self.model2.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop.')
                break

        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        total_explicit, correct_explicit = 0, 0
        total_implicit, correct_implicit = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model2.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_batch['polarity'].to(self.opt.device)

                t_outputs = self.model2(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

                pred = torch.argmax(t_outputs, -1)
                if self.opt.task == 'absa':
                    implicits = t_batch['implicit'].to(self.opt.device)
                    total_explicit += (~implicits).long().sum().item()
                    correct_explicit += ((~implicits) & (pred == t_targets)).long().sum().item()
                    total_implicit += implicits.long().sum().item()
                    correct_implicit += (implicits & (pred == t_targets)).long().sum().item()

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
        if self.opt.task == 'absa':
            explicit_acc = correct_explicit / total_explicit if total_explicit else 0.0
            implicit_acc = correct_implicit / total_implicit if total_implicit else 0.0
            return acc, f1, explicit_acc, implicit_acc
        else:
            return acc, f1

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params2 = filter(lambda p: p.requires_grad, self.model2.parameters())
        optimizer2 = self.opt.optimizer2(_params2, lr=self.opt.lr, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        best_model_path = self._train2(criterion, optimizer2, train_data_loader, val_data_loader)
        self.model2.load_state_dict(torch.load(best_model_path))
        if self.opt.task == 'absa':
            test_acc, test_f1, ese, ise = self._evaluate_acc_f1(test_data_loader)
            logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}, explicit acc: {:.4f}, implicits acc: {:.4f}'
                        .format(test_acc, test_f1, ese, ise))
        else:
            test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
            logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name1', default='bert_spc1', type=str)
    parser.add_argument('--model_name2', default='bert_spc', type=str)
    parser.add_argument('--aug_multi', default='1', type=int, help='1, 2, 3, 4')
    parser.add_argument('--task', default='absa', type=str, help='isa, absa')
    parser.add_argument('--dataset', default='semrest_edarsrd1', type=str, help='twitter, restaurant, laptop, entity1')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch_alpha', default=10, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--num_epoch', default=10, type=int, help='try larger number for non-BERT models')
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
    parser.add_argument('--beta2', default=0.1, type=float)
    parser.add_argument('--aug', default='all', type=str,
                        help='xxx_insert, bert_substitute, bert_insert, word2vec_insert')
    parser.add_argument('--aug_ratio', default='0.1', type=str, help='0.1, 0.2, 0.3, 0.4')
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

    opt.model_class2 = model_classes[opt.model_name2]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name2]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer2 = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = '{}-{}-{}.log'.format(opt.model_name2, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
