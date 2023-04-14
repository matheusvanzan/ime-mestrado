#!/usr/bin/env python
# coding: utf-8


import os
import evaluate

from statistics import mean
from webbrowser import get
from accelerate.accelerator import AcceleratedOptimizer
from collections import Counter
from sklearn.metrics import confusion_matrix
from pprint import pprint

import torch
from transformers.models import gpt2
torch.manual_seed(42)

from datetime import datetime
from torch.utils.data import random_split
from transformers import GPT2Tokenizer, GPT2TokenizerFast, MobileViTForImageClassification, TrainingArguments, \
    Trainer, GPT2LMHeadModel, GPT2ForSequenceClassification, EarlyStoppingCallback, TrainerCallback

import pandas as pd
import numpy as np

import whats
import settings
from dataset import DirectoryDataset


class GPT:

    def __init__(self, model_name, model_label, limit, fold, epochs, batch):
        print(f'Model: name {model_name}')
        print(f"Model: label '{model_label}'")        

        self.chunk_size = settings.GPT_TRAIN_CHUNK_SIZE

        self.model_name = model_name
        self.model_label = model_label
        self.limit = limit
        self.fold = fold
        self.epochs = epochs
        self.batch_size = batch

        if self.model_label == 'all':
            self.num_labels = len(settings.DATASET_CLASSES)
        else:
            self.num_labels = 2
        print(f'Model: num_labels {self.num_labels}')

        path = os.path.join(settings.PATH_DATA, model_name)        

        # Ex: 5.limit-256.chunk-16.epochs-2.batch-16
        path_label_name = '{}.limit-{}.fold-{}.chunk-{}.epochs-{}.batch-{}'.format(
            self.model_label,
            self.limit,
            self.fold,
            self.chunk_size,
            self.epochs,
            self.batch_size
        )
        path_label = os.path.join(path, path_label_name)        

        self.output_dir = os.path.join(path_label, 'partial')
        self.logging_dir = os.path.join(path_label, 'logs')
        self._create_paths()

        self.path_tokenizer = os.path.join(path_label, 'tokenizer')
        self.path_model = os.path.join(path_label, 'model')
        self.path_model_trained = os.path.join(path_label, 'model-trained')

        self.path_cmatrix = os.path.join(path_label, 'cmatrix.csv')
        self.path_results = os.path.join(path_label, 'results.csv')        

        self.model = None
        self.tokenizer = self._get_tokenizer()

        self.bos_token_id = self.tokenizer('<|startoftext|>').input_ids[0] # 50257
        self.eos_token_id = self.tokenizer('<|endoftext|>').input_ids[0] # 50256
        self.pad_token_id = self.tokenizer('<|pad|>').input_ids[0] # 50258  

    def _create_paths(self):

        paths = [
            self.output_dir,
            self.logging_dir,
        ]

        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

    def _get_tokenizer(self):

        if not os.path.exists(self.path_tokenizer):
            print('Model: downloading tokenizer...')
            tokenizer = GPT2Tokenizer.from_pretrained(
                self.model_name,
                num_labels = self.num_labels
            )
            tokenizer.pad_token = tokenizer.eos_token
            print('Model: saving tokenizer...')
            torch.save(tokenizer, self.path_tokenizer)
            print('Model: tokenizer saved!')
        else:
            print('Model: using saved tokenizer...')
            tokenizer = torch.load(self.path_tokenizer)

        return tokenizer

    def _get_model_original(self):
        print('Model: looking for', self.path_model)

        if not os.path.exists(self.path_model):
            print('Model: downloading model...')
            model = GPT2ForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels = self.num_labels
            )
            model.config.pad_token_id = model.config.eos_token_id
            print('Model: caching model...')
            torch.save(model, self.path_model)
            print('Model: model cached!')
        else:
            print('Model: using cached model...')
            model = torch.load(self.path_model)

        model.to(torch.device('cuda'))

        return model

    def _get_model_trained(self):

        print(f'Model: using trained at {self.path_model_trained} ...')
        model = GPT2ForSequenceClassification.from_pretrained(
            self.path_model_trained
        )
        # model.config.pad_token_id = model.config.eos_token_id
        model.to(torch.device('cuda'))
        return model

    def _data_collator(self, data):
        return {
            'input_ids': torch.stack([f[1] for f in data]),       
            'attention_mask': torch.stack([f[2] for f in data]),
            'labels': torch.stack([f[3] for f in data])
        }

    def trained(self):
        return os.path.isfile(
            os.path.join(self.path_model_trained, 'config.json')
        )

    def train(self, train_dataset, val_dataset):

        model = self._get_model_original()

        training_args = TrainingArguments(
            output_dir = self.output_dir,
            logging_dir = self.logging_dir,

            # evaluation_strategy = 'steps',
            # eval_steps = 10,

            logging_strategy = 'steps',
            logging_steps = 1000,

            save_strategy = 'steps',
            save_steps = 1000,

            num_train_epochs = self.epochs,
                                           
            per_device_train_batch_size = self.batch_size,
            per_device_eval_batch_size = self.batch_size,

            warmup_steps = 0,
            weight_decay = 0.01,

            # metric_for_best_model = 'accuracy',
            # load_best_model_at_end = True,

            report_to = 'tensorboard'
        )

        # def compute_metrics(eval_pred):
        #     print('\n\n\n***** Compute Metrics *****\n\n\n')
        #     return None

        class StepCallback(TrainerCallback):
            def on_log(self, args, state, control, **kwargs):
                # if state.global_step % 5 == 0:
                print('Calling StepCallback.on_log()')
                msg = 'GPT-2 Training\n---------------\n'
                msg += '- epochs: {}/{}\n'.format(round(state.epoch, 2), state.num_train_epochs)
                msg += '- steps: {}/{}\n'.format(state.global_step, state.max_steps)
                msg += '- percent: {}%'.format(round(100*state.global_step/state.max_steps, 2))
                whats.send(msg)


        trainer = Trainer(
            model = model, 
            args = training_args,
            train_dataset = train_dataset,
            eval_dataset = val_dataset, 
            data_collator = self._data_collator,

            # compute_metrics = compute_metrics,
            callbacks = [StepCallback()]
        )

        try:
          train = trainer.train(resume_from_checkpoint = True)
        except:
          train = trainer.train()
        
        print('Model: saving trained model...')
        trainer.save_model(self.path_model_trained)
        print('Model: model saved!')
        return train.metrics

    def results(self):
        rv = []
        with open(self.path_results, 'r') as f:
            for line in f.readlines():
                items = line.split(',')
                rv.append([ items[0], int(items[1]) ])
        return rv

    def metrics(self):
        print(self.path_results)

        predictions = []
        references = []
        good = 0
        bad = 0
        with open(self.path_results, 'r') as f:
            for line in list(f.readlines())[1:]:
                items = line.split(',')
                pred = int(items[1].strip())
                ref = int(items[2].strip())
                predictions.append(pred)
                references.append(ref)

                if pred == ref:
                    good += 1
                else:
                    bad += 1

        print('GOOD:', good)
        print('BAD:', bad)

        cm = confusion_matrix(references, predictions)
        print(cm)

        m = {}
        m.update(evaluate.load('accuracy').compute(
            predictions = predictions, 
            references = references
        ))
        # m.update(evaluate.load('precision').compute(
        #     predictions = predictions, 
        #     references = references,
        #     average = None
        # ))
        # m.update(evaluate.load('recall').compute(
        #     predictions = predictions, 
        #     references = references,
        #     average = None
        # ))
        m.update(evaluate.load('f1').compute(
            predictions = predictions, 
            references = references,
            average = None
        ))
        return m

    def average(self, l, w = None):
        '''
            l = [0, 1, 0, 0]
            average = 0.25
            returns 0
        '''
        if not w:
            return sum([int(x) for x in l]) / len(l)

        else:
            return sum(int(x)*int(y) for x, y in zip(l, w)) / sum(w)

    def test(self, test_dataset):
        
        model_trained = self._get_model_trained()
        trainer = Trainer(
            model = model_trained,
            data_collator = self._data_collator
        )

        # metrics by chunk
        test_predictions_weights = trainer.predict(test_dataset)
        # test_predictions = np.argmax(test_predictions_weights[0], axis=1)

        test_labels = np.array([label for _, _, _, label in test_dataset])
        test_references = np.array(test_labels)

        # metrics by file
        files = {}
        test_predictions_2 = []
        test_references_2 = []
        for i in range(len(test_dataset)):
            file_id, _, _, label = test_dataset[i]

            predw = test_predictions_weights[0][i]
            # pred = test_predictions[i]
            # print('predw', predw)

            pred = np.argmax(predw)
            # print('pred', pred)

            dist = max(predw[0], predw[1]) - min(predw[0], predw[1])
            # print('dist', dist)

            if file_id not in files:
                files.update({ file_id:{'pred': [], 'dist': [], 'ref': []} })

            files[file_id]['pred'].append(pred)
            files[file_id]['dist'].append(dist)
            files[file_id]['ref'].append(int(label))

        with open(self.path_results, 'w+') as f:
            f.write(f"file_id,pred_{self.model_label},ref\n")

            for file_id, values in files.items():
                ref = values['ref'][0]

                # counter
                pred_3c = Counter(values['pred'])
                pred_3 = pred_3c.most_common()[0][0]

                f.write(f"{file_id},{pred_3},{ref}\n")

        return self.metrics()















