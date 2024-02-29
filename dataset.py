import os
import torch
torch.manual_seed(42)
import pickle
import pandas as pd
import itertools

from torch.utils.data import Dataset, random_split
from sys import getsizeof

import settings

def leakage_check(l1, l2, l3):
    # print('leakage_check')
    # print(len(l1), len(l2), len(l3))

    i12 = set.intersection(set(l1), set(l2))
    i13 = set.intersection(set(l1), set(l3))
    i23 = set.intersection(set(l2), set(l3))
    
    if len(i12) + len(i13) + len(i23) > 0:
        raise Exception('DATA LEAK')

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

def custom_one_hot_encoder(n, total):
    return [0]*n + [1] + [0]*(total-n-1)

class DirectoryDataset(Dataset):

    def __init__(self, tokenizer, label, label_0, path, split_name, list_dir, limit, fold, chunk, version, use_cache):

        self.verbose = False

        self.tokenizer = tokenizer
        self.label = label
        self.label_0 = label_0
        self.path = os.path.join(path, 'by-label', label)        
        self.split_name = split_name
        self.list_dir = list_dir
        self.limit = limit
        self.fold = fold
        self.version = version

        self.use_cache = use_cache

        self.chunk = chunk
        self.chunks = []

        # multiclass
        if self.label_0 == 'all':
            self.label_id = torch.tensor ( int(self.label) )

        # binary
        else:
            if str(self.label_0) == str(self.label): # if ref model
                self.label_id = torch.tensor( 0 )
            else:
                self.label_id = torch.tensor( 1 )

        if self.verbose:
            print(f'Dataset: label {self.label} - {self.split_name} - in model as torch.tensor({self.label_id})')
            print('self.path', self.path)

        # directory
        path_csv_parent = os.path.join(path, f'version-{version}', f'fold-{self.fold}')
        if not os.path.exists(path_csv_parent):
            os.makedirs(path_csv_parent)        
        
        # file name
        self.path_csv = os.path.join(path_csv_parent, f'{label}.limit-{self.limit}.fold-{self.fold}.chunk-{self.chunk}.version-{self.version}.{self.split_name}.csv')

        self._create_csv() # if not in file already
        self._create_input_ids()

    def _tokenize_chunks(self, content, filename):
        '''
            content: f.read() of all files in the Directory
            we can limit here files content (by word count or token count)
        '''
        tokens = self.tokenizer(content)
        # tokens = {'input_ids': [...], 'attention_mask': [...]}

        start_token = self.tokenizer(self.tokenizer.bos_token)
        end_token = self.tokenizer(self.tokenizer.eos_token)
        pad_token = self.tokenizer(self.tokenizer.pad_token)

        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']

        # if len(input_ids) > self.limit:
        #     print(f'Warning: truncating ({len(input_ids)} > {self.limit})')

        #### TEMP
        # len_ = len(input_ids)
        # # print('len(input_ids)', self.label, self.split_name, filename, len_)
        # with open(f'tokens.csv', 'a+') as f:
        #     f.write(f'{self.label},{self.split_name},{filename},{len_}\n')
        ### END TEMP

        if self.limit != 'all':
            input_ids = input_ids[:self.limit]
            attention_mask = attention_mask[:self.limit]

        input_id_chunks = get_chunks(
            list_ = input_ids, # start_token['input_ids'] + input_ids + end_token['input_ids'],
            window = self.chunk, 
            fill = pad_token['input_ids'][0]
        )

        mask_chunks = get_chunks(
            list_ = attention_mask, # start_token['attention_mask'] + attention_mask + end_token['attention_mask'],
            window = self.chunk, 
            fill = pad_token['attention_mask'][0]
        )

        ### CAN'T PRINT list(generator)
        ### it consumes the generator
        ### TODO: fix -> 

        return input_id_chunks, mask_chunks

    def _create_csv(self):

        if self.use_cache and os.path.isfile(self.path_csv):
            if self.verbose:
                print(f'Dataset: label {self.label} - {self.split_name} - loading from {self.path_csv} ...')

        else:
            if self.verbose:
                print(f'Dataset: label {self.label} - {self.split_name} - creating from {self.path} ...')
                print(f'Dataset: label {self.label} - {self.split_name} - {len(self.list_dir)} files')

            # create .csv
            with open(self.path_csv, 'w+') as file_csv:
                pass

            # loop .asm files
            len_list_dir = len(self.list_dir)
            for i, filename in enumerate(self.list_dir):                
                filepath = os.path.join(self.path, filename)
                fileid = filename.split('.')[0]

                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                # write chunks
                with open(self.path_csv, 'a') as file_csv:
                    input_id_chunks, mask_chunks = self._tokenize_chunks(content, filename)
                    for input_ids, mask in zip(input_id_chunks, mask_chunks):                    
                        line = fileid + ','
                        line += ','.join([str(i) for i in input_ids + mask])
                        line += '\n'
                        file_csv.write(line)

                if len_list_dir > 10 and i % (int(len_list_dir/10)) == 0:
                    if self.verbose:
                        print(f'Dataset: label {self.label} - {self.split_name} - {int(100*i/len_list_dir)}%')

            if self.verbose:
                print(f'Dataset: label {self.label} - {self.split_name} - saved at {self.path_csv}')

    def _create_input_ids(self):

        with open(self.path_csv, 'r') as f:
            self.chunks = f.readlines()

        if self.verbose:
            print(f'Dataset: label {self.label} - {self.split_name} - {len(self.chunks)} chunks - {getsizeof_gb(self.chunks)} Gb')

    def getfile(self, idx):
        chunk = self.chunks[idx]
        chunk_list = [x for x in chunk.split(',')]
        file_id = chunk_list[0]
        return file_id

    def __len__(self):
        if not self.tokenizer:
            raise Exception('tokenizer not set')

        return len(self.chunks)

    def __getitem__(self, idx):
        if not self.tokenizer:
            raise Exception('tokenizer not set')

        chunk = self.chunks[idx]
        chunk_list = [x for x in chunk.split(',')]

        file_id = chunk_list[0]
        chunk_int = [int(x) for x in chunk_list[1:]]

        input_ids = torch.tensor( chunk_int[:self.chunk] )
        attn_mask = torch.tensor( chunk_int[self.chunk:] )
        
        return file_id, input_ids, attn_mask, self.label_id


class DatasetHelper:

    def __init__(self, tokenizer, labels, label_0, path, limit, fold, chunk, version, use_cache=True):

        self.verbose = False

        self.tokenizer = tokenizer
        self.labels = labels
        self.label_0 = label_0
        self.path = path # proc-1
        self.limit = limit
        self.fold = fold
        self.chunk = chunk
        self.use_cache = use_cache

        if version in [1, 2]:
            self.version = version
        else:
            self.version = 3
        
        self.list_dir = {
            'train': [None]*len(labels),
            'eval' : [None]*len(labels),
            'test' : [None]*len(labels)
        }

        for i, label in enumerate(self.labels):

            list_dir_path = os.path.join(self.path, 'by-label', label)

            if self.verbose:
                print('----\nlabel', label)
                print('- version', self.version)
                print('- list_dir_path', list_dir_path)

            # point to ovewrite os.listdir for versioning
            os_list_dir = self._os_listdir(list_dir_path)

            len_10 = int(0.1 * len(os_list_dir))
            len_80 = len(os_list_dir) - 2*len_10

            # print('NO K-FOLD')    
            # self.list_dir['train'][i], \
            # self.list_dir['eval'][i], \
            # self.list_dir['test'][i] = random_split(os_list_dir, [len_80, len_10, len_10])

            if self.verbose:
                print('os_list_dir', len(os_list_dir), '-', os_list_dir[0], '...', os_list_dir[-1])        
            
            # WITH K-FOLD
            splits = random_split(os_list_dir, [len(os_list_dir) - 9*len_10] + 9*[len_10])
            # print('splits', splits[0][0], splits[-1][0])

            # print('WITH K-FOLD')
            fold_0 = self.fold - 1

            splits_shifted = splits[fold_0:] + splits[:fold_0]
            # print('splits_shifted', splits_shifted[0][0], splits_shifted[-1][0])

            splits_train = [ splits_shifted[i] for i in range(8) ]
            splits_eval = splits_shifted[8]
            splits_test = splits_shifted[9]

            self.list_dir['train'][i] = list(itertools.chain.from_iterable( splits_train ))
            self.list_dir['eval'][i]  = splits_eval
            self.list_dir['test'][i]  = splits_test

            if self.verbose:
                for split in ['train', 'eval', 'test']:
                    print('  ', split, '-', len(self.list_dir[split][i]), '-', self.list_dir[split][i][0], '...', self.list_dir[split][i][-1])

            leakage_check(
                self.list_dir['train'][i],
                self.list_dir['eval'][i],
                self.list_dir['test'][i]
            )


    def _os_listdir(self, path):
        '''
            Sorting os.listdir()
        '''
        print('===')
        print('os.listdir ovewrite', path)
        
        list_dir = os.listdir(path)

        if int(self.version) in [1, 2]: # os defined (old)
            print('list_dir')
            print('  - first', list_dir[0:2])
            print('  - last', list_dir[-3:])
            return list_dir

        if int(self.version) >= 3: # sorted
            print('---')
            list_dir_sorted = sorted(list_dir)
            print('list_dir_sorted')
            print('  - first', list_dir_sorted[0:2])
            print('  - last', list_dir_sorted[-3:])
            return list_dir_sorted

        return None

    def _get_dataset(self, split_name):
        # tokenizer, label, path, split_name, list_dir, use_cache

        dataset = None
        for i, label in enumerate(self.labels):
            dd = DirectoryDataset(
                tokenizer = self.tokenizer,
                label = label,
                label_0 = self.label_0,
                path = self.path,
                split_name = split_name, 
                list_dir = self.list_dir[split_name][i],
                limit = self.limit,
                fold = self.fold,
                chunk = self.chunk,
                version = self.version,
                use_cache = self.use_cache
            )

            if not dataset:
                dataset = dd
            else:
                dataset += dd

        return dataset

    def get_train(self):
        return self._get_dataset('train')

    def get_eval(self):
        return self._get_dataset('eval')

    def get_test(self):
        return self._get_dataset('test')






