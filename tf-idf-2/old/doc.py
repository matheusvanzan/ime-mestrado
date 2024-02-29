'''

Parse all docs inside given directory
Read each line and remove junk chars


'''

import os
import psutil
import pickle
import pandas as pd
import datetime
import functools
import operator

from math import sqrt

from functools import reduce
from sklearn.feature_extraction.text import CountVectorizer
from concurrent.futures import ProcessPoolExecutor
from collections import Counter

from mem import check_mem


def create_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def save_counter(d, path):
    start = datetime.datetime.now()
    mode = os.path.splitext(path)[-1]

    path_dir = os.path.dirname(path)
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)

    if mode == '.pkl':
        with open(path, 'wb') as f:
            pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

    if mode == '.txt':
        with open(path, 'w+', encoding='utf-8', errors='ignore') as f:
            for key, value in d.items():
                f.write(f'{key},{value}\n')
    
    stop = datetime.datetime.now()
    # print(f'time to write {path} {stop-start}')

def load_counter(path):
    start = datetime.datetime.now()
    mode = os.path.splitext(path)[-1]
    vocab = {}

    path_dir = os.path.dirname(path)
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)

    if mode == '.pkl':
        with open(path, 'rb') as f:
            vocab = pickle.load(f)

    if mode == '.txt':
        with open(path, 'r+', encoding='utf-8', errors='ignore') as f:
            for line in f.read().split('\n'):
                items = line.split(',')
                if len(items) == 2:
                    vocab.update({items[0]: int(items[1])})
    
    stop = datetime.datetime.now()
    # print(f'time to load {path} {stop-start}')

    return Counter(vocab)


class DocManager:

    def __init__(self, docs_limit, path_data_raw, path_data_counts, \
        processor, max_features, ngram, max_workers):

        self.docs_limit = docs_limit
        self.path_data_raw = path_data_raw
        self.path_data_counts = path_data_counts
        self.processor = processor
        self.max_features = max_features
        self.ngram = ngram
        self.max_workers = max_workers

    def set_ngram(self, ngram):
        self.ngram = ngram

    
    def create_count_from_asm(self, filetype='.asm'):
        list_dir = os.listdir(self.path_data_raw)
        if len(list_dir) < self.docs_limit:
            self.docs_limit = len(list_dir)

        values = []
        for i, filename in enumerate(list_dir):
            if i+1 > self.docs_limit: break
            values.append((i+1, self.docs_limit, filename))

        # Singleprocessor
        # for value in values:
        #     self.create_count_from_single_asm(value)

        # Multiprocessor
        with ProcessPoolExecutor(max_workers = self.max_workers) as executor:
            for i, vocab in enumerate(executor.map( \
                    self.create_count_from_single_asm, values)):
                vocab # use it just to execute ProcessPoolExecutor

    def create_count_from_single_asm(self, values):
        start = datetime.datetime.now()

        i, total, filename = values

        path_asm = os.path.join(self.path_data_raw, filename)
        path_count = os.path.join(self.path_data_counts, str(self.ngram), filename.replace('.asm', '.txt'))
        
        if os.path.exists(path_count):
            print(f'{i}/{total}: skip {filename}')
            vocab = Counter({})

        else:
            print(f'{i}/{total}: start {filename}')

            with open(path_asm, encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            content = self.processor.process(content)
            vocab = Counter({})

            words = content.strip().split(' ')
            if self.ngram == 1:
                vocab = Counter(words)
            else:
                for i in range(len(words) - self.ngram + 1):
                    token = ' '.join([words[i+j] for j in range(self.ngram)])
                    if not token in vocab:
                        vocab.update({ token : 0 })
                    vocab[token] += 1

            save_counter(vocab, path_count)
            stop = datetime.datetime.now()
            print(f'{i}/{total} finish - vocab {len(vocab)} - time {stop-start}')

        check_mem()
        return vocab

    def create_vocab_from_files(self):
        path = os.path.join(self.path_data_counts, f'{self.ngram}')
        list_dir = sorted(os.listdir(path))
        if len(list_dir) < self.docs_limit:
            self.docs_limit = len(list_dir)
        
        values = []
        for i, filename in enumerate(list_dir):
            if i+1 > self.docs_limit: break
            values.append((i+1, self.docs_limit, filename))


        # acm
        chunk_size = 100 # int(sqrt(self.docs_limit))
        print('chunk size', chunk_size)
        chunks = list(create_chunks(values, chunk_size))
        vocab_accumulated = Counter({})

        # Singleprocessor
        # for chunk in chunks:
        #     vocab_accumulated += self.accumulate_counts(chunk)

        # Multiprocessor
        with ProcessPoolExecutor(max_workers = self.max_workers) as executor:
            for vocab in executor.map(self.accumulate_counts, chunks):
                vocab_accumulated += vocab

        # save to file count/accumulated
        path = os.path.join(self.path_data_counts, f'vocab.{self.docs_limit}.{self.ngram}.txt')
        save_counter(vocab_accumulated, path)

        return vocab_accumulated
    
    def accumulate_counts(self, values, use_cache=False):

        if use_cache:
            i_0 = values[0][0]
            i_n = values[-1][0]
            path_vocab = os.path.join(
                f'{self.path_data_counts}',
                f'vocab.p{i_0}-{i_n}.{self.docs_limit}.{self.ngram}.txt'
            )

            if os.path.exists(path_vocab):
                print(f'skip {path_vocab}')
                check_mem()
                return load_counter(path_vocab)

        vocab_accumulated = Counter({})
        for value in values:
            i, total, filename = value
            print(f'{i}/{total} - start {filename}')
            path = os.path.join(self.path_data_counts, f'{self.ngram}')
            vocab = load_counter(os.path.join(path, f'{filename}'))
            vocab_accumulated += vocab
            print(f'{i}/{total} - acm', list(vocab_accumulated.items())[:3])
            check_mem()

        if use_cache:
            save_counter(vocab_accumulated, path_vocab)

        return vocab_accumulated
