import os
import torch
torch.manual_seed(42)
import pickle
import pandas as pd

from torch.utils.data import Dataset, random_split
from sys import getsizeof

import settings


def get_chunks(list_, window):
    '''
    args:
        list_ = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        window = 2
    return:
        [(0, 1), (2, 3), (4, 5), (6, 7)]
    '''
    # tuples = list(zip(*(iter(list_),) * window))
    # return [list(t) for t in tuples]

    for i in range(0, len(list_), window):
        if i + window < len(list_):
            yield list_[i:i + window]

def getsizeof_gb(x):
    return round(getsizeof(x)/(1024*1024), 2)


class TextDataset(Dataset):

    def __init__(self, tokenizer, contents):
        self.input_ids = []
        self.attn_masks = []

        self.chunk_size = settings.DATASET_CHUNK_SIZE

        self.tokenizer = tokenizer
        self.contents = contents

        print('Dataset ->', len(self.contents), 'files')

        input_id_start = self.tokenizer('<|startoftext|>').input_ids
        input_id_end = self.tokenizer('<|endoftext|>').input_ids

        mask_start = self.tokenizer('<|startoftext|>').attention_mask
        mask_end = self.tokenizer('<|endoftext|>').attention_mask

        total_chunks = 0
        for i, content in enumerate(self.contents):
            # print(i, '- len', len(content))

            input_ids, mask = self.tokenize_chunks(content)
            self.input_ids.append(input_ids)
            self.attn_masks.append(mask)

            len_chunk = int(len(content)/self.chunk_size)

            if len_chunk == 0:
                print(i, 'erro 0 chunks')

            if i % 100 == 0:
                print(i, '-', len_chunk, 'chunks')

            total_chunks += len_chunk

        print('Total chunks -', total_chunks)

    def tokenize_chunks(self, content):

        ## start end on each chunk
        # tokens = tokenizer(content)

        # input_id_chunks = get_chunks(tokens['input_ids'], self.chunk_size - 2)
        # for input_id in input_id_chunks:
        #     input_id = input_id_start + input_id + input_id_end # 1024 - 2 ('<|startoftext|>' and '<|endoftext|>')
        #     self.input_ids.append(torch.tensor(input_id))

        # mask_chunks = get_chunks(tokens['attention_mask'], self.chunk_size - 2)
        # for mask in mask_chunks:
        #     mask = mask_start + mask + mask_end # 1024 - 2 ('<|startoftext|>' and '<|endoftext|>')
        #     self.attn_masks.append(torch.tensor(mask))

        ## start end on all content
        tokens = self.tokenizer('<|startoftext|>' + content + '<|endoftext|>') 
        # tokens = {'input_ids': [...], 'attention_mask': [...]}

        input_id_chunks = get_chunks(tokens['input_ids'], self.chunk_size)
        # for input_id in input_id_chunks:
        #     self.input_ids.append(torch.tensor(input_id))

        mask_chunks = get_chunks(tokens['attention_mask'], self.chunk_size)
        # for mask in mask_chunks:
        #     self.attn_masks.append(torch.tensor(mask))

        for input_id, mask in zip(input_id_chunks, mask_chunks):
            yield (
                torch.tensor(input_id),
                torch.tensor(mask)
            )


class DirectoryDataset(Dataset):

    def __init__(self, tokenizer, path, limit):
        self.tokenizer = tokenizer
        self.path = path
        self.limit = limit

        self.chunk_size = settings.DATASET_CHUNK_SIZE
        self.chunks = []

        self.split_names = ['train', 'val', 'test']

        # define names
        self.list_dir = list(os.listdir(path))
        self.paths_csv = [
            path + f'.train.csv',
            path + f'.val.csv',
            path + f'.test.csv'
        ]

        # split 80 / 10 / 10
        #    test / val / train
        _10pc = int(0.1 * len(self.list_dir))
        self.sizes = [len(self.list_dir) - 2*_10pc, _10pc, _10pc]
        self.list_dirs = list(random_split(self.list_dir, self.sizes))

        # for split in ['train', 'val', 'test']:
        #     print(split, sizes[split], self.list_dirs[split][0])       

        print('Dataset: {} {} files'.format( \
            len(self.list_dir), self.sizes ))

        self._create_csv() # if not in file already

        self._create_input_ids()        

    def _tokenize_chunks(self, content):
        tokens = self.tokenizer('<|startoftext|>' + content + '<|endoftext|>') 
        # tokens = {'input_ids': [...], 'attention_mask': [...]}

        input_id_chunks = get_chunks(tokens['input_ids'], self.chunk_size)
        mask_chunks = get_chunks(tokens['attention_mask'], self.chunk_size)
        return zip(input_id_chunks, mask_chunks)

    def _create_csv(self):

        # check if already exists
        if all([os.path.isfile(x) for x in self.paths_csv]):
            for path_csv in self.paths_csv:
                print('Dataset: loading from', path_csv, '...')


        # if not, create
        else:
            print('Dataset: creating from', self.path, '...')

            for path_csv, list_dir in zip(self.paths_csv, self.list_dirs):

                # create .csv
                with open(path_csv, 'w+') as file_csv:
                    pass

                # loop .asm files
                for i, filename in enumerate(list_dir):
                    filepath = os.path.join(self.path, filename)

                    with open(filepath, 'r') as f:
                        content = f.read()

                    # write chunk
                    for input_ids, mask in self._tokenize_chunks(content):
                        with open(path_csv, 'a') as file_csv:
                            line = ','.join([str(i) for i in input_ids + mask])
                            line += '\n'
                            file_csv.write(line)

                    if i % 100 == 0:
                        print('Dataset:', i, '- len', len(content))

                print('Dataset: saved!')

    def _create_input_ids(self):

        with open(self.path_csv, 'r') as f:
            self.chunks = f.readlines()

        print('Dataset:', len(self.chunks), 'chunks')
        print('Dataset:', getsizeof_gb(self.chunks), 'Gb in memmory!')

    def __len__(self):
        if not self.tokenizer:
            raise Exception('tokenizer not set')

        return len(self.chunks)

    def __getitem__(self, idx):
        if not self.tokenizer:
            raise Exception('tokenizer not set')

        chunk = self.chunks[idx]
        chunk_int = [int(x) for x in chunk.split(',')]

        input_ids = torch.tensor( chunk_int[:self.chunk_size] )
        attn_mask = torch.tensor( chunk_int[self.chunk_size:] )
        return input_ids, attn_mask

    def train_val_test_split(self):
        '''
            90 % - train + val
                 81 % - train
                  9 % - val
            10 % - test
        '''

        test_size = int(0.1 * len(self))
        val_size = int(0.09 * len(self))
        train_size = len(self) - test_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            self, [train_size, val_size, test_size])

        # print('train_dataset', len(train_dataset))
        # print('val_dataset', len(val_dataset))
        # print('test_dataset', len(test_dataset))

        return train_dataset, val_dataset, test_dataset


def get_dataset2(tokenizer):
    descriptions = pd.read_csv(settings.PATH_PROJECT + 'netflix_titles.csv')['description'][:50] # limit 10

    # max_length = max([len(tokenizer.encode(description)) for description in descriptions])
    # print('max_length', max_length)

    dataset = TextDataset(descriptions)
    dataset.set_tokenizer(tokenizer)
    return dataset





