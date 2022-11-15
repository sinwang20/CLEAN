# -*- coding: utf-8 -*-
# file: data_preprocess.py

import argparse
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, RobertaTokenizer
import random
import nlpaug.augmenter.word as naw
import json

from eda import *


# for the first time you use wordnet, nlpaug
# import nltk
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')


def gen_eda(train_orig, output_file, alpha_rs, alpha_rd, alpha_ss=0.1, alpha_ri=0.1, aug_multi=1, stopwords=None):
    writer = open(output_file, 'w')
    lines = open(train_orig, 'r', encoding='utf-8').readlines()

    for i in range(0, len(lines), 3):
        sentence = lines[i].lower().strip()
        sentence_7mw2 = sentence.replace("$t$", "7mw2")

        aspect = lines[i + 1].lower().strip()
        polarity = lines[i + 2].strip()

        writer.write(sentence_7mw2 + '\n')

        # eda-random swap
        aug_sentences_swap = eda(sentence, alpha_sr=0, alpha_ri=0, alpha_rs=alpha_rs, p_rd=0, num_aug=aug_multi)

        for aug_sentence in aug_sentences_swap:
            aug_sentence = aug_sentence.replace("$t$", "7mw2")
            # print(aug_sentence)
            writer.write(aug_sentence + '\n')
        # eda-random delete
        aug_sentences_delete = eda(sentence, alpha_sr=0, alpha_ri=0, alpha_rs=0, p_rd=alpha_rd, num_aug=aug_multi)
        for aug_sentence in aug_sentences_delete:
            aug_sentence = aug_sentence.replace("$t$", "7mw2")
            writer.write(aug_sentence + '\n')

        # nlpaug
        # wordnet synonym-substitute
        aug_wordnet = naw.SynonymAug(aug_src='wordnet', aug_max=10, aug_p=alpha_ss, stopwords=stopwords)
        aug_sentences_substitue = aug_wordnet.augment(sentence_7mw2, n=aug_multi)
        # print(aug_sentences_substitue)
        if aug_multi > 1:
            print('aug-multi')
            for aug_sentence in aug_sentences_substitue:
                # print(aug_sentence)
                writer.write(aug_sentence + '\n')
        else:
            # print(aug_sentences_substitue)
            writer.write(aug_sentences_substitue + '\n')
        # w2v random-insert
        aug_w2v = naw.WordEmbsAug(model_type='word2vec', aug_max=10, aug_p=alpha_ri, stopwords=stopwords
                                  , action="insert", model_path="/root/pack/GoogleNews-vectors-negative300.bin")
        aug_sentences_insert = aug_w2v.augment(sentence_7mw2, n=aug_multi)
        if aug_multi > 1:
            for aug_sentence in aug_sentences_insert:
                writer.write(aug_sentence + '\n')
        else:
            # print(aug_sentences_insert)
            writer.write(aug_sentences_insert + '\n')

        writer.write(aspect + '\n')
        writer.write(polarity + '\n')

    writer.close()
    print("generated augmented sentences with eda for " + train_orig + " to " + output_file + " with aug_multi=" + str(
        aug_multi))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default='../datasets/semeval14/Laptops_Train.xml.seg', type=str,
                        help="input file of unaugmented data")
    parser.add_argument("--output", default='../datasets/semeval14/semlaptop_new_1.txt', type=str,
                        help="output file of unaugmented data")
    parser.add_argument('--task', default='absa', type=str, help='isa, absa')
    parser.add_argument("--aug_multi", default=1, type=int,
                        help="number of augmented sentences per original sentence per augmentation method, like 1, 2, 4, 8")
    # in our work CLEAN, we just simply set the percent alpha as 0.1
    parser.add_argument("--alpha_rs", default=0.1, type=float, help="percent of words in each sentence to be swapped")
    parser.add_argument("--alpha_rd", default=0.1, type=float, help="percent of words in each sentence to be deleted")
    parser.add_argument("--alpha_ss", default=0.1, type=float,
                        help="percent of words in each sentence to be substituted")
    parser.add_argument("--alpha_ri", default=0.1, type=float, help="percent of words in each sentence to be inserted")
    args = parser.parse_args()

    # since the former special token '$Y$' for the aspect will be splited in nlpaug,
    # so we choose a strange word '7mw2', which doesn't appear in the corpus, to avoid the aspect word(s) being damaged.
    # It's also the homonym of the traditional Chinese monster '魑魅魍魉‘, so Gook Luck for you！
    stopwords = None
    if args.task == 'absa':
        stopwords = ['7mw2']

    gen_eda(args.input, args.output,
            alpha_rs=args.alpha_rs,
            alpha_rd=args.alpha_rd,
            alpha_ss=args.alpha_ss,
            alpha_ri=args.alpha_ri,
            aug_multi=args.aug_multi,
            stopwords=stopwords)
