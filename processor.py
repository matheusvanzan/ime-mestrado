#!/usr/bin/env python
# coding: utf-8

import settings

import os
import datetime
import shutil
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


class Processor:

    def __init__(self):
        self.docs_limit = 100000 # settings.DOCS_LIMIT
        self.chars_to_remove = settings.NPL_CHARS_TO_REMOVE
        self.words_to_remove = settings.NPL_WORDS_TO_REMOVE
        self.path_data_labels = settings.PATH_DATA_LABELS
        self.path_data_raw = settings.PATH_DATA_ASM
        self.path_data_proc = settings.PATH_DATA_PROC_1
        self.vocab = settings.NPL_VOCAB
        self.max_workers = 2 # settings.MAX_WORKERS

    def filter_segment(self, content, segment='pure code'):
        '''
            CODE    -       Pure code   <-------- what we want
            DATA    -       Pure data
            CONST   -       Pure data
            BSS     -       Uninitialized data
            STACK   -       Uninitialized data
            XTRN    -       Extern definitions segment
        '''

        segments = content.split('segment type')
        content_segments = ''

        for seg in segments:
            if segment in seg[:10].lower():
                # print(len(seg.split(' ')), '|', seg[:30])
                content_segments += ' ' + seg.strip()

        return content_segments

    def filter_vocab(self, content):
        words = content.split(' ')
        n = len(words)
        return_words = []

        for i in range(n):
            w = words[i]
            w_left = words[i-1] if i>0 else None
            w_right = words[i+1] if i<n-1 else None

            if w in self.vocab:
                return_words.append(w)
            elif len(w) > 2 and (w_left in self.vocab or w_right in self.vocab):
                return_words.append(w)

        return ' '.join(return_words)

    def process(self, content):
        # lower case
        content = content.lower()

        # remove chars
        for char_to_remove in self.chars_to_remove:
            content = content.replace(char_to_remove, ' ')

        # remove words
        for word_to_remove in self.words_to_remove:
            content = content.replace(word_to_remove, ' ')

        # remove multiple whitespaces
        content = ' '.join(content.split())

        # keep only code segment
        content = self.filter_segment(content)

        # only vocab
        if len(self.vocab) != 0:
            content = self.filter_vocab(content)

        return content.strip()

    def process_all_docs(self):
        list_dir = os.listdir(self.path_data_raw)
        if len(list_dir) < self.docs_limit:
            self.docs_limit = len(list_dir)

        values = []
        for i, filename in enumerate(list_dir):
            if i+1 > self.docs_limit: break
            values.append((i+1, self.docs_limit, filename))

        # Singleprocessor
        for value in values:
            self.process_single_doc(value)

        # Multiprocessor
        #with ProcessPoolExecutor(max_workers = self.max_workers) as executor:
        #    for i, x in enumerate(executor.map( \
        #            self.process_single_doc, values)):
        #            x # use it just to execute ProcessPoolExecutor

    def process_single_doc(self, values):
        start = datetime.datetime.now()

        i, total, filename = values

        path_raw = os.path.join(self.path_data_raw, filename)
        path_proc = os.path.join(self.path_data_proc, 'all', filename)
        
        if os.path.exists(path_proc):
            pass #print(f'{i}/{total}: skip {filename}')

        else:
            print(f'{i}/{total}: start {filename}')

            content = None
            with open(path_raw, encoding='utf-8', errors='replace') as f:
                content = f.read()
                content = self.process(content)

            with open(path_proc, 'w+', encoding='utf-8') as f:
                f.write(content)
            
            stop = datetime.datetime.now()
            print(f'{i}/{total} finish - time {stop-start}')

        return ''

    def split_by_label(self):
        print('split_by_label')

        content = pd.read_csv(self.path_data_labels)
        for id_, label in zip(content['Id'], content['Class']):
            old_path = os.path.join(self.path_data_proc, 'all', f'{id_}.asm')
            new_path = os.path.join(self.path_data_proc, str(label), f'{id_}.asm')
            print(old_path)
            print(new_path)
            shutil.copyfile(old_path, new_path)
            print('--')












