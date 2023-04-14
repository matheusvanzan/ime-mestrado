#!/usr/bin/env python
# coding: utf-8


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import settings
import datetime
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)

from sklearn.metrics import confusion_matrix

import torch
torch.manual_seed(42)

from torch.utils.data import random_split
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPT2LMHeadModel, GPTNeoForCausalLM
from sentence_transformers import SentenceTransformer, util
from argparse import ArgumentParser
from pprint import pprint

from processor import Processor
from gpt import GPT
from dataset import DatasetHelper
from metrics import get_metrics


def main(args):

    if args.process:

        proc = Processor()
        # proc.process_all_docs()
        # proc.split_by_label()
        proc.sanity_check()

    else:

        limit = settings.DATASET_LIMIT
        if args.limit:
            limit = 'all' if args.limit == 'all' else int(args.limit)

        epochs = settings.GPT_EPOCHS
        if args.epochs:
            epochs = int(args.epochs)

        batch = settings.GPT_BATCH_SIZE
        if args.batch:
            batch = int(args.batch)

        labels = settings.DATASET_CLASSES
        if args.label:
            labels = [args.label]

        path = settings.PATH_DATA_PROC_1
        if args.path:
            path = args.path

        model_name = settings.MODEL_NAME
        if args.model:
            model_name = args.model

        multi = False
        if args.multi:
            multi = True
            labels = ['all']

        fold = 1
        if args.fold:
            fold = int(args.fold)

        print('Args: limit', limit)
        print('Args: epochs', epochs)
        print('Args: batch', batch)
        print('Args: labels', labels)
        print('Args: path', path)
        print('Args: model_name', model_name)
        print('Args: multi', multi)
        print('Args: fold', fold)

        # CACHE
        if args.cache:
            gpt = GPT(
                model_name=model_name, 
                model_label=labels[0], 
                limit=limit,
                fold = fold,
                epochs=epochs, 
                batch=batch
            )

            helper = DatasetHelper(
                tokenizer = gpt.tokenizer,
                labels = labels,
                label_0 = labels[0],
                path = path, 
                limit = limit,
                fold = fold,
                use_cache = True
            )
            
            train_dataset = helper.get_train()
            val_dataset = helper.get_eval()
            test_dataset = helper.get_test()


        # TRAIN
        if args.train:
            
            results = {}
            for label_0 in labels:
                print('START', label_0)
                gpt = GPT(
                    model_name=model_name, 
                    model_label=label_0, 
                    limit=limit,
                    fold = fold,
                    epochs=epochs,
                    batch=batch
                )

                if not gpt.trained():
                    helper = DatasetHelper(
                        tokenizer = gpt.tokenizer,
                        labels = settings.DATASET_CLASSES, # todos independente da entrada
                        label_0 = label_0,
                        path = path, 
                        limit = limit,
                        fold = fold,
                        use_cache = True
                    )
                    train_dataset = helper.get_train()
                    val_dataset = helper.get_eval()
                    metrics = gpt.train(train_dataset, val_dataset)
                    results.update({label_0: metrics})
                print('END', label_0)
                
            pprint(results)

        # CHECK e TEST
        if args.check or args.test:
            
            results = {}
            for label_0 in labels:
                gpt = GPT(
                    model_name=model_name, 
                    model_label=label_0, 
                    limit=limit,
                    fold = fold,
                    epochs=epochs,
                    batch=batch
                )

                helper = DatasetHelper(
                    tokenizer = gpt.tokenizer,
                    labels = settings.DATASET_CLASSES, # todos independente da entrada
                    label_0 = label_0,
                    path = path, 
                    limit = limit,
                    fold = fold,
                    use_cache = True
                )
                test_dataset = helper.get_test()
                gpt.test(test_dataset)
                metrics = gpt.metrics()
                results.update({label_0: metrics})
            pprint(results)

        # METRICS
        if args.metrics:
            
            metrics = {}
            # results = []
            for label_0 in labels:
                gpt = GPT(model_name=model_name, model_label=label_0, limit=limit, epochs=epochs, batch=batch)
                metric = gpt.metrics()
                metrics.update({label_0: metric})

                # r = gpt.results()
                # results.append(r)

            pprint(metrics)

            # columns = ['label_' + x for x in settings.DATASET_CLASSES] + ['pred', 'ref']
            # index = []
            # rows = []
            # refs = {}

            # with open(settings.PATH_DATA_LABELS, 'r') as f:
            #     for line in list(f.readlines())[1:]:
            #         line = line.replace('"', '').strip()
            #         items = line.split(',')
            #         refs.update({items[0]: int(items[1])-1})

            # for i in range(1, len(results[0])): # file id 0..1081
            #     file_id_0, pred_0 = results[0][i]
            #     index.append(file_id_0)

            #     row = []
            #     for j in range(len(results)): # labels 0..8
            #         file_id, pred = results[j][i]
            #         row.append(pred)

            #         if file_id != file_id_0:
            #             print('error', file_id, file_id_0)

            #     row.append(np.argmin(row))
            #     row.append(refs[file_id])
            #     rows.append(row)

            # df = pd.DataFrame(rows, index=index, columns=columns)
            # # print(df)
            
            # for i in range(10):
            #     df_tmp = df[df.drop('ref', axis=1).sum(axis=1) == i]
            #     if len(df_tmp) > 0:
            #         print('Soma', i)
            #         print(df_tmp)

            # matrix = confusion_matrix(df['ref'], df['pred'])
            # print(matrix)

            # metrics = get_metrics(df['ref'], df['pred'])
            # print(metrics)
        



if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('-p', '--process', dest='process', required=False, action='store_true', help='Process dataset')
    parser.add_argument('-c', '--complete', dest='complete', required=False, action='store_true', help='Complete prompt')
    parser.add_argument('-t', '--train', dest='train', required=False, action='store_true', help='Train the models')
    parser.add_argument('-ck', '--check', dest='check', required=False, action='store_true', help='Check the models')
    parser.add_argument('-te', '--test', dest='test', required=False, action='store_true', help='Test the models')
    parser.add_argument('-mm', '--metrics', dest='metrics', required=False, action='store_true', help='Get model metrics')
    parser.add_argument('-cc', '--cache', dest='cache', required=False, action='store_true', help='Dataset cache')

    parser.add_argument('-mu', '--multi', dest='multi', required=False, action='store_true', help='Multiclass training')

    parser.add_argument('-mo', '--model', dest='model', required=False, help='Set model')
    parser.add_argument('-li', '--limit', dest='limit', required=False, help='Dataset token limit')
    parser.add_argument('-eo', '--epochs', dest='epochs', required=False, help='Model  epochs')
    parser.add_argument('-ba', '--batch', dest='batch', required=False, help='Model  size')
    parser.add_argument('-la', '--label', dest='label', required=False, help='Dataset label (class)')
    parser.add_argument('-fo', '--fold', dest='fold', required=False, help='Dataset k-fold')

    parser.add_argument('-pa', '--path', dest='path', required=False, help='Dataset path')
    
    
    args = parser.parse_args()
    print(args)

    print(f'pid: {os.getpid()}')
    # input('press any key to continue...')

    start = datetime.datetime.now()
    # pre()
    main(args)
    stop = datetime.datetime.now()
    print(str(stop-start).split('.')[0])


