'''
-src_data 'data/english.txt' -trg_data 'data/french.txt' -src_lang english -trg_lang french

'''

from utils_W import *


def train_model(train, src_pad, trg_pad):
    for epoch in range(1):
        for i, batch in enumerate(train):
            print('\n')
            # print('batch.src:', batch.src)
            src = batch.src.transpose(0, 1)
            print('src:', src)

            trg = batch.trg.transpose(0, 1)
            print('trg:', trg)

            trg_input = trg[:, :-1]
            print('trg_input:', trg_input)
            src_mask, trg_mask = create_masks(src, trg_input, src_pad, trg_pad)
            print('src_mask:', src_mask)
            print('trg_mask:', trg_mask)


src_data, trg_data = read_dataAll('./data/my_en.txt', './data/my_fr.txt')
print('type(src_data):', type(src_data))
print('src_data:', src_data)
print('trg_data:', trg_data)

SRC, TRG = create_fieldsAll('en', 'fr')
src_pad, trg_pad, train_len, train = create_datasetAll(max_strlen=30, batchsize=5,
                                                       src_data=src_data, source=SRC, trg_data=trg_data, target=TRG)
print('src_pad:', src_pad)
print('trg_pad:', trg_pad)
train_model(train, src_pad, trg_pad)
