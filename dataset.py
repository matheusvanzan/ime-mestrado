import os
import torch
torch.manual_seed(42)
import pickle
import pandas as pd

from torch.utils.data import Dataset, random_split
from sys import getsizeof

import settings


def get_chunks(list_, window, fill=None, limit=None):
    '''
    args:
        list_ = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        window = 2
        limit = None
    return:
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, None]]

        limit = 3
        [[0, 1], [2, 3], [4, 5]]
    '''
    end = len(list_)

    if limit:
        end = min(end, limit)

    for i in range(0, end, window):
        if i + window < len(list_):
            yield list_[i:i + window]
        else:
            rv = list_[i:i + window]
            yield (window - len(rv))*[fill] + rv # padding left

def getsizeof_gb(x):
    return round(getsizeof(x)/(1024*1024), 2)


class DirectoryDataset(Dataset):

    def __init__(self, tokenizer, path, split_name, list_dir, use_cache):

        self.limit = settings.DATASET_LIMIT
        self.tokenizer = tokenizer
        self.path = path
        self.split_name = split_name
        self.list_dir = list_dir

        self.use_cache = use_cache

        self.chunk_size = settings.DATASET_CHUNK_SIZE
        self.chunks = []

        self.path_csv = f'{path}.{split_name}.limit-{self.limit}.chunk-{self.chunk_size}.csv'

        self._create_csv() # if not in file already
        self._create_input_ids()

    def _tokenize_chunks(self, content):
        '''
            content: f.read() of all files in the Directory

            here we can limit files content (by word count or token count)
        '''
        tokens = self.tokenizer(content)
        # tokens = {'input_ids': [...], 'attention_mask': [...]}

        # TODO: change limit to here
        # TODO: add optional first and last words in get_chunks

        start_token = self.tokenizer('<|startoftext|>')
        end_token = self.tokenizer('<|endoftext|>')
        pad_token = self.tokenizer('<|pad|>')

        input_id_chunks = get_chunks(
            list_ = start_token['input_ids'] + tokens['input_ids'][:self.limit] + end_token['input_ids'],
            window = self.chunk_size, 
            fill = pad_token['input_ids'][0],
            limit = self.limit
        )

        mask_chunks = get_chunks(
            list_ = start_token['attention_mask'] + tokens['attention_mask'][:self.limit] + end_token['attention_mask'],
            window = self.chunk_size, 
            fill = pad_token['attention_mask'][0],
            limit = self.limit
        )

        ### CAN'T PRINT list(generator)
        ### it consumes the generator
        ### TODO: fix -> 

        return input_id_chunks, mask_chunks

    def _create_csv(self):

        if self.use_cache and os.path.isfile(self.path_csv):
            print(f'Dataset: {self.split_name} - loading from {self.path_csv} ...')

        else:
            print(f'Dataset: {self.split_name} - creating from {self.path} ...')
            print(f'Dataset: {self.split_name} - {len(self.list_dir)} files')

            # create .csv
            with open(self.path_csv, 'w+') as file_csv:
                pass

            # loop .asm files
            len_list_dir = len(self.list_dir)
            for i, filename in enumerate(self.list_dir):
                filepath = os.path.join(self.path, filename)

                with open(filepath, 'r') as f:
                    content = f.read()

                # write chunks
                input_id_chunks, mask_chunks = self._tokenize_chunks(content)
                for input_ids, mask in zip(input_id_chunks, mask_chunks):
                    with open(self.path_csv, 'a') as file_csv:
                        line = ','.join([str(i) for i in input_ids + mask])
                        line += '\n'
                        file_csv.write(line)

                if len_list_dir > 10 and i % (int(len_list_dir/10)) == 0:
                    print(f'Dataset: {self.split_name} - {int(100*i/len_list_dir)}%')

            print(f'Dataset: {self.split_name} - saved at {self.path_csv}')

    def _create_input_ids(self):

        with open(self.path_csv, 'r') as f:
            self.chunks = f.readlines()

        print(f'Dataset: {self.split_name} - {len(self.chunks)} chunks - {getsizeof_gb(self.chunks)} Gb')

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



class DatasetHelper:

    def __init__(self, tokenizer, label, path, limit, use_cache=True):

        self.tokenizer = tokenizer
        self.path = path
        self.use_cache = use_cache

        list_dir = os.listdir(path)
        len_10 = int(0.1 * len(list_dir))
        len_80 = len(list_dir) - 2*len_10

        # if limit:
        #     self.list_dir = self.list_dir[:self.limit]
        #     self.path_csv = path + f'.limit-{limit}.csv'

        self.train_list_dir, \
        self.eval_list_dir, \
        self.test_list_dir = random_split(list_dir, [len_80, len_10, len_10])


    def _get_dataset(self, split):
        return DirectoryDataset(
            self.tokenizer, 
            self.path, 
            split, 
            self.train_list_dir, 
            self.use_cache
        )

    def get_train(self):
        return self._get_dataset('train')

    def get_eval(self):
        return self._get_dataset('eval')

    def get_test(self):
        return self._get_dataset('test')

def get_dataset(tokenizer, label, path, limit, use_cache=True):
    
    list_dir = os.listdir(path)
    len_10 = int(0.1 * len(list_dir))
    len_80 = len(list_dir) - 2*len_10

    # if limit:
    #     self.list_dir = self.list_dir[:self.limit]
    #     self.path_csv = path + f'.limit-{limit}.csv'

    train_list_dir, eval_list_dir, test_list_dir = \
        random_split(list_dir, [len_80, len_10, len_10])

    train_dataset = DirectoryDataset(tokenizer, path, 'train', train_list_dir, use_cache)
    eval_dataset  = DirectoryDataset(tokenizer, path, 'eval',  eval_list_dir, use_cache)
    test_dataset  = DirectoryDataset(tokenizer, path, 'test',  test_list_dir, use_cache)

    return train_dataset, eval_dataset, test_dataset


class CompletionDataset:
    '''
        name: dataset-1-model-1.csv

    '''

