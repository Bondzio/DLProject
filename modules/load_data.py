"""

# Sparse Sequence-to-Sequence Models

load_data.py

"""


PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
BEGIN_TOKEN = '<begin>'
END_TOKEN = '<end>'

PAD_ID = 0
UNK_ID = 1
BEGIN_ID = 2
END_ID = 3




import numpy as np
import pandas as pd
import math

from tqdm import tqdm, tnrange, tqdm_notebook
from time import time


import os


from typing import List, Tuple, Dict, Set, Union


import codecs
import re


import torch
import torchvision



def remove_multiple_whitespaces(s):
      return re.sub(' +', ' ', s)

def load_data_file(folder, name):
    data = []
    filepath = os.path.join(folder, name)
    with codecs.open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='%s lines' % name):
            line = line.strip()
            line = remove_multiple_whitespaces(line.strip()).split(' ')
            data.append(line)
    return np.array(data)



class Vocabulary:
    def __init__(self, name=None):
        self.name = name
        self.token2id = {
            PAD_TOKEN : PAD_ID,
            UNK_TOKEN : UNK_ID,
            BEGIN_TOKEN : BEGIN_ID,
            END_TOKEN : END_ID
        }
        self.id2token = {v : k for k, v in self.token2id.items()}
        self.pad_id = self.token2id[PAD_TOKEN]
        self.unk_id = self.token2id[UNK_TOKEN]
        self.begin_id = self.token2id[BEGIN_TOKEN]
        self.end_id = self.token2id[END_TOKEN]

        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN
        self.begin_token = BEGIN_TOKEN
        self.end_token = END_TOKEN
    
    def __len__(self):
        return len(self.token2id)
    
    def __contains__(self, token):
        return token in self.token2id

    def __getitem__(self, token):
        return self.token2id.get(token, self.unk_id)
    
    def tokens2ids(self, tokens):
        if type(tokens[0]) == list:
            return [[self[t] for t in l] for l in tokens]
        return [self[t] for t in tokens]

    def ids2tokens(self, ids):
        if type(ids[0]) == list:
            return [[self.id2token[i] for i in l] for l in ids]
        return [self.id2token[i] for i in ids]
    
    def add_token(self, token):
        if token not in self:
            token_id = self.token2id[token] = len(self)
            self.id2token[token_id] = token

    def add_tokens(self, tokens):
        for token in tokens:
            self.add_token(token)
    
    def prepare_for_model(self, tokens: List[List[str]], device=None):
        # to ints:
        ids = self.tokens2ids(tokens)

        # add padding and transpose
        max_len = max(len(l) for l in ids)
        batch_size = len(ids)

        transposed_ids = []
        for i in range(max_len):
            transposed_ids.append([ids[k][i] if len(ids[k]) > i else self.pad_id for k in range(batch_size)])
        
        # transform to torch.Tensor [max_len, batch_size]
        tensor = torch.tensor(transposed_ids, dtype=torch.long, device=device)

        return tensor


def make_vocabulary_inflection(from_file, vocab_type, vocab_name):
    vocab = Vocabulary(vocab_name)
    if vocab_type == 'from':
        tokens = set()
        tag_tokens = set()
        for line in tqdm(from_file, desc='making "%s" vocabulary from lines:' % vocab_name):
            for token in line:
                if len(token) > 1 and token[0] == '_':
                    tag_tokens.add(token)
                else:
                    tokens.add(token)
        vocab.add_tokens(sorted(list(tag_tokens)))
        vocab.add_tokens(sorted(list(tokens)))
    elif vocab_type == 'to':
        tokens = set()
        for line in tqdm(from_file, desc='making "%s" vocabulary from lines:' % vocab_name):
            for token in line:
                tokens.add(token)
        vocab.add_tokens(sorted(list(tokens)))
    print("\nSize of '%s' vocabulary: %d\n" % (vocab_name, len(vocab)))
    return vocab


def make_vocabulary_mt(file_path, vocab_name):
    vocab = Vocabulary(vocab_name)
    tokens = set()
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc='making "%s" vocabulary from lines:' % vocab_name):
            tokens.add(line.split(' ')[0])
    vocab.add_tokens(sorted(list(tokens)))
    print("\nSize of '%s' vocabulary: %d\n" % (vocab_name, len(vocab)))
    return vocab




