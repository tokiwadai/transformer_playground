from torchtext import data
import numpy as np
from torch.autograd import Variable

from Tokenize import *
import torch
import pandas as pd
from Batch import *
import os


def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)),
                      k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    return np_mask


def create_masks(src, trg, src_pad, trg_pad):
    src_mask = (src != src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != trg_pad).unsqueeze(-2)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size)
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None
    return src_mask, trg_mask


def read_data(src_data_path):
    try:
        return open(src_data_path).read().strip().split('\n')
    except:
        print("error: '" + src_data_path + "' file not found")
        quit()


def read_dataAll(src_data_path, trg_data_path):
    if src_data_path is not None:
        try:
            src_data = open(src_data_path).read().strip().split('\n')
        except:
            print("error: '" + src_data_path + "' file not found")
            quit()

    if trg_data_path is not None:
        try:
            trg_data = open(trg_data_path).read().strip().split('\n')
        except:
            print("error: '" + trg_data_path + "' file not found")
            quit()
    return src_data, trg_data


def create_fields(src_lang):
    spacy_langs = ['en', 'fr', 'de', 'es', 'pt', 'it', 'nl']
    if src_lang not in spacy_langs:
        print('invalid src language: ' + src_lang + 'supported languages : ' + spacy_langs)

    print("loading spacy tokenizers...")
    t_src = tokenize(src_lang)
    print("finished spacy tokenizers...")
    src1 = data.Field(lower=True, tokenize=t_src.tokenizer, pad_token='<pad>')

    return src1


def create_fieldsAll(src_lang, trg_lang):
    spacy_langs = ['en', 'fr', 'de', 'es', 'pt', 'it', 'nl']
    if src_lang not in spacy_langs:
        print('invalid src language: ' + src_lang + 'supported languages : ' + spacy_langs)
    if trg_lang not in spacy_langs:
        print('invalid trg language: ' + trg_lang + 'supported languages : ' + spacy_langs)

    print("loading spacy tokenizers...")
    t_src = tokenize(src_lang)
    t_trg = tokenize(trg_lang)

    TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>', pad_token='<pad>')
    SRC = data.Field(lower=True, tokenize=t_src.tokenizer, pad_token='<pad>', pad_first=True)

    return SRC, TRG


def get_len(train):
    for i, b in enumerate(train):
        pass
    return i


class MyIterator1(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


global max_src_in_batch1


def batch_size_fn1(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch1
    if count == 1:
        max_src_in_batch1 = 0
    max_src_in_batch1 = max(max_src_in_batch1, len(new.src))
    src_elements = count * max_src_in_batch1
    # print('type(src_elements):', type(src_elements))
    # print('src_elements:', src_elements)
    return max(src_elements, 0)


def create_dataset(max_strlen, batchsize, device, src_data1, SRC1):
    print("creating dataset and iterator... ")
    raw_data = {'src': [line for line in src_data1]}
    df = pd.DataFrame(raw_data, columns=["src"])

    mask = (df['src'].str.count(' ') < max_strlen)
    df = df.loc[mask]

    df.to_csv("./data/translate_transformer_temp.csv", index=False)

    data_fields = [('src', SRC1)]
    train = data.TabularDataset('./data/translate_transformer_temp.csv', format='csv', fields=data_fields)
    SRC1.build_vocab(train)

    train_iter1 = MyIterator1(train, batch_size=batchsize, device=device,
                              repeat=False, sort_key=lambda x: len(x.src),
                              batch_size_fn=batch_size_fn1, train=True, shuffle=True)

    os.remove('./data/translate_transformer_temp.csv')

    src_pad1 = SRC1.vocab.stoi['<pad>']
    train_len1 = get_len(train_iter1)

    return src_pad1, train_len1, train_iter1


def create_datasetAll(max_strlen, batchsize, src_data, source, trg_data, target):
    print("creating dataset and iterator... ")

    raw_data = {'src': [line for line in src_data], 'trg': [line for line in trg_data]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])

    mask = (df['src'].str.count(' ') < max_strlen) & (df['trg'].str.count(' ') < max_strlen)
    df = df.loc[mask]

    df.to_csv("./data/translate_transformer_temp.csv", index=False)

    data_fields = [('src', source), ('trg', target)]
    train = data.TabularDataset('./data/translate_transformer_temp.csv', format='csv', fields=data_fields)
    source.build_vocab(train)
    target.build_vocab(train)

    train_iter = MyIterator(train, batch_size=batchsize,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True, shuffle=True)

    os.remove('./data/translate_transformer_temp.csv')

    src_pad = source.vocab.stoi['<pad>']
    trg_pad = target.vocab.stoi['<pad>']

    train_len = get_len(train_iter)

    return src_pad, trg_pad, train_len, train_iter
