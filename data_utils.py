# -*- coding: utf-8 -*-
# file: data_utils.py


import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import random
import json


def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    # print(maxlen)
    # if len(sequence) > 85:
    #    print("new_length", len(sequence))
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        print(pretrained_bert_name)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)  # BertTokenizer
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()

            text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            context_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            left_indices = tokenizer.text_to_sequence(text_left)
            left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            right_with_aspect_indices = tokenizer.text_to_sequence(aspect + " " + text_right, reverse=True)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_len = np.sum(left_indices != 0)
            aspect_len = np.sum(aspect_indices != 0)
            aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)
            polarity = int(polarity) + 1

            text_len = np.sum(text_indices != 0)
            concat_bert_indices = tokenizer.text_to_sequence(
                '[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
            concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
            concat_segments_indices = pad_and_truncate(concat_segments_indices, tokenizer.max_seq_len)

            text_bert_indices = tokenizer.text_to_sequence(
                "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
            aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

            data = {
                'concat_bert_indices': concat_bert_indices,
                'concat_segments_indices': concat_segments_indices,
                'text_bert_indices': text_bert_indices,
                'aspect_bert_indices': aspect_bert_indices,
                'text_indices': text_indices,
                'context_indices': context_indices,
                'left_indices': left_indices,
                'left_with_aspect_indices': left_with_aspect_indices,
                'right_indices': right_indices,
                'right_with_aspect_indices': right_with_aspect_indices,
                'aspect_indices': aspect_indices,
                'aspect_boundary': aspect_boundary,
                'polarity': polarity,
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSADataset6(Dataset):
    def __init__(self, fname, aug_multi, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []

        textedanum = (3 + aug_multi * 4)  # 5,7
        for i in range(0, len(lines), textedanum):  # 7 5
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("7mw2")]
            aspect = lines[i + textedanum - 2].lower().strip()  # 5
            polaritys = lines[i + textedanum - 1].strip()  # 6

            concat_bert_text_bias = []
            concat_segments_indice_bias = []
            text_indices_bias = []
            aspect_indices_bias = []
            left_indices_bias = []
            text_bert_indices_bias = []
            aspect_bert_indices_bias = []

            text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            text_len = np.sum(text_indices != 0)
            left_indices = tokenizer.text_to_sequence(text_left)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            aspect_len = np.sum(aspect_indices != 0)

            polarity = int(polaritys)

            concat_bert_indices = tokenizer.text_to_sequence(
                '[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
            concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
            concat_segments_indices = pad_and_truncate(concat_segments_indices, tokenizer.max_seq_len)

            text_bert_indices = tokenizer.text_to_sequence(
                "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
            aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

            concat_bert_text_bias.append(concat_bert_indices)
            concat_segments_indice_bias.append(concat_segments_indices)
            text_indices_bias.append(text_indices)
            aspect_indices_bias.append(aspect_indices)
            left_indices_bias.append(left_indices)
            text_bert_indices_bias.append(text_bert_indices)
            aspect_bert_indices_bias.append(aspect_bert_indices)

            # eda wordnet w2v_insert
            for j in range(4 * aug_multi):  # 1,2
                texti_left, _, texti_right = [s.lower().strip() for s in lines[i + 1 + j].partition("7mw2")]
                texti_indices = tokenizer.text_to_sequence(texti_left + " " + aspect + " " + texti_right)
                texti_len = np.sum(texti_indices != 0)
                lefti_indices = tokenizer.text_to_sequence(texti_left)

                concat_bert_indices_xi = tokenizer.text_to_sequence(
                    '[CLS] ' + texti_left + " " + aspect + " " + texti_right + ' [SEP] ' + aspect + " [SEP]")
                concat_segments_indices_xi = [0] * (texti_len + 2) + [1] * (aspect_len + 1)
                concat_segments_indices_xi = pad_and_truncate(concat_segments_indices_xi, tokenizer.max_seq_len)

                texti_bert_indices = tokenizer.text_to_sequence(
                    "[CLS] " + texti_left + " " + aspect + " " + texti_right + " [SEP]")

                concat_bert_text_bias.append(concat_bert_indices_xi)
                concat_segments_indice_bias.append(concat_segments_indices_xi)
                text_indices_bias.append(texti_indices)
                aspect_indices_bias.append(aspect_indices)
                left_indices_bias.append(lefti_indices)
                text_bert_indices_bias.append(texti_bert_indices)
                aspect_bert_indices_bias.append(aspect_bert_indices)

            concat_bert_text_bias = torch.tensor(concat_bert_text_bias)
            concat_segments_indice_bias = torch.tensor(concat_segments_indice_bias)
            text_indices_bias = torch.tensor(text_indices_bias)
            aspect_indices_bias = torch.tensor(aspect_indices_bias)
            left_indices_bias = torch.tensor(left_indices_bias)
            text_bert_indices_bias = torch.tensor(text_bert_indices_bias)
            aspect_bert_indices_bias = torch.tensor(aspect_bert_indices_bias)

            data = {
                'concat_bert_indices': concat_bert_text_bias,
                'concat_segments_indices': concat_segments_indice_bias,
                'text_bert_indices': text_bert_indices_bias,
                'aspect_bert_indices': aspect_bert_indices_bias,
                'text_indices': text_indices_bias,
                'aspect_indices': aspect_indices_bias,
                'left_indices': left_indices_bias,
                'polarity': polarity,
            }
            all_data.append(data)

        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def str_to_bool(str):
    return True if str == 'True' else False


class ABSADataset7(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        for i in range(0, len(lines), 4):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            implicit = lines[i + 2].strip()
            polarity = lines[i + 3].strip()

            text = text_left + " " + aspect + " " + text_right
            text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            context_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            left_indices = tokenizer.text_to_sequence(text_left)
            left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            right_with_aspect_indices = tokenizer.text_to_sequence(aspect + " " + text_right, reverse=True)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_len = np.sum(left_indices != 0)
            aspect_len = np.sum(aspect_indices != 0)
            aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)
            polarity = int(polarity)
            implicit = str_to_bool(implicit)

            text_len = np.sum(text_indices != 0)
            concat_bert_indices = tokenizer.text_to_sequence(
                '[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
            concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
            concat_segments_indices = pad_and_truncate(concat_segments_indices, tokenizer.max_seq_len)

            text_bert_indices = tokenizer.text_to_sequence(
                "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
            aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

            data = {
                'concat_bert_indices': concat_bert_indices,
                'concat_segments_indices': concat_segments_indices,
                'text_bert_indices': text_bert_indices,
                'aspect_bert_indices': aspect_bert_indices,
                'text_indices': text_indices,
                'context_indices': context_indices,
                'left_indices': left_indices,
                'left_with_aspect_indices': left_with_aspect_indices,
                'right_indices': right_indices,
                'right_with_aspect_indices': right_with_aspect_indices,
                'aspect_indices': aspect_indices,
                'aspect_boundary': aspect_boundary,
                'polarity': polarity,
                'implicit': implicit,
                'text': text,
                'aspect': aspect
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def polarity_to_num(str):
    if str == 'positive':
        return 2
    elif str == 'negative':
        return 0
    else:
        return 1


class ISADataset1(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        for i in range(0, len(lines), 2):
            text = lines[i].lower().strip()
            polarity = lines[i + 1].strip()

            text_indices = tokenizer.text_to_sequence(text)
            concat_bert_indices = tokenizer.text_to_sequence("[CLS] " + text + " [SEP]")

            polarity = int(polarity)

            data = {
                'concat_bert_indices': concat_bert_indices,
                'text_indices': text_indices,
                'polarity': polarity,
            }
            all_data.append(data)

        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ISADataset6(Dataset):
    def __init__(self, fname, aug_multi, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        textedanum = (2 + aug_multi * 4)  # 5,7
        for i in range(0, len(lines), textedanum):  # 7 5
            text = lines[i].lower().strip()
            polarity = lines[i + textedanum - 1].strip()

            concat_bert_text_bias = []

            concat_bert_indices = tokenizer.text_to_sequence("[CLS] " + text + " [SEP]")
            concat_bert_text_bias.append(concat_bert_indices)
            # eda
            for j in range(4 * aug_multi):
                texti = lines[i + 1 + j].lower().strip()
                concat_bert_indices_xi = tokenizer.text_to_sequence("[CLS] " + texti + " [SEP]")
                concat_bert_text_bias.append(concat_bert_indices_xi)

            concat_bert_text_bias = torch.tensor(concat_bert_text_bias)

            polarity = int(polarity)

            data = {
                'concat_bert_indices': concat_bert_text_bias,
                'polarity': polarity
            }

            all_data.append(data)

        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
