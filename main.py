#!/usr/bin/env python
# coding: utf-8


import os
import settings
import datetime

import pandas as pd

from torch.utils.data import random_split
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPT2LMHeadModel, GPTNeoForCausalLM
from sentence_transformers import SentenceTransformer, util
from argparse import ArgumentParser

from processor import Processor
from gpt import GPT, GPTOriginal, GPTTester
from dataset import DatasetHelper


def main(args):

    if args.process:

        proc = Processor()
        # proc.process_all_docs()
        # proc.split_by_label()
        proc.sanity_check()

    if args.complete:

        prompt = 'The key benefits of TF-IDF are'

        # GTP
        gpt_original = GPTOriginal(model_name=settings.MODEL_NAME)
        completion = gpt_original.complete(prompt)
        print('-----------------')
        print(prompt)
        print('-----------------')
        print(completion)
        print('-----------------')

    else:

        labels = settings.DATASET_CLASSES
        if args.label:
            labels = [args.label]

        path = settings.PATH_DATA_PROC_1
        if args.path:
            path = args.path

        # CACHE
        if args.cache:

             for i in labels:
                label = str(i)

                gpt = GPT(model_name=settings.MODEL_NAME, model_label=label)
                helper = DatasetHelper(
                    tokenizer = gpt.tokenizer,
                    label = label,
                    path = os.path.join(path, label),
                    limit = settings.DATASET_LIMIT,
                    use_cache = False
                )


        # TRAIN
        if args.train:
            for i in labels:
                label = str(i)

                gpt = GPT(model_name=settings.MODEL_NAME, model_label=label)

                helper = DatasetHelper(
                    tokenizer = gpt.tokenizer,
                    label = label,
                    path = os.path.join(path, label), 
                    limit = settings.DATASET_LIMIT,
                    use_cache = True
                )
                train_dataset = helper.get_train()
                val_dataset = helper.get_eval()
                helper.get_test() # just to create it
                gpt.train(train_dataset, val_dataset)

        # CHECK e TEST
        if args.check or args.test:

            gpt_list = []
            dataset_list = []

            for i in labels:
                label = str(i)

                gpt = GPT(model_name=settings.MODEL_NAME, model_label=label)
                gpt_list.append(gpt)

                helper = DatasetHelper(
                    tokenizer = gpt.tokenizer,
                    label = label, 
                    path = os.path.join(path, label), 
                    limit = settings.DATASET_LIMIT,
                    use_cache = True
                )
                test_dataset = helper.get_test()
                dataset_list.append(test_dataset)

            gpt_tester = GPTTester(gpt_list, dataset_list)

            if args.check:
                gpt_tester.check()

            if args.test:
                df, acc = gpt_tester.test()
                pd.set_option('max_columns', None)
                print(df)
                print('Acc', acc)

        



if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('-p', '--process', dest='process', required=False, action='store_true', help='Process dataset')
    parser.add_argument('-c', '--complete', dest='complete', required=False, action='store_true', help='Complete prompt')
    parser.add_argument('-t', '--train', dest='train', required=False, action='store_true', help='Train the models')
    parser.add_argument('-ck', '--check', dest='check', required=False, action='store_true', help='Check the models')
    parser.add_argument('-e', '--test', dest='test', required=False, action='store_true', help='Test the models')

    parser.add_argument('-l', '--label', dest='label', required=False, help='Dataset label (class)')
    parser.add_argument('-pa', '--path', dest='path', required=False, help='Dataset path')
    parser.add_argument('-cc', '--cache', dest='cache', required=False, action='store_true', help='Dataset cache')
    
    
    args = parser.parse_args()
    print(args)

    print(f'pid: {os.getpid()}')
    # input('press any key to continue...')

    start = datetime.datetime.now()
    # pre()
    main(args)
    stop = datetime.datetime.now()
    print(str(stop-start).split('.')[0])


