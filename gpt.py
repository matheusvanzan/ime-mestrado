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

        self.early_stop = False

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
        self.path_train_metrics = os.path.join(path_label, 'train-metrics.json')

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

            num_train_epochs = self.epochs,

            # log for tensorboard
            logging_strategy = 'steps',
            logging_steps = 1000,

            # save for continuig training
            save_strategy = 'steps',
            save_steps = 10000,                      
                                           
            per_device_train_batch_size = self.batch_size,
            per_device_eval_batch_size = self.batch_size,

            warmup_steps = 0,
            weight_decay = 0.01,

            # earlystopping
            evaluation_strategy = 'steps' if self.early_stop else 'no',
            eval_steps = 100,
            load_best_model_at_end = self.early_stop,
            metric_for_best_model = 'accuracy',            

            report_to = 'tensorboard'
        )

        def compute_metrics(eval_pred):
            print('\nCompute Metrics\n')
            # print('eval_pred', eval_pred)

            # print('eval_pred.label_ids', eval_pred.label_ids)
            # print('eval_pred.predictions', eval_pred.predictions)

            chunk_predictions_weights = eval_pred.predictions
            chunk_predictions = np.argmax(chunk_predictions_weights, axis=1)
            # print('chunk_predictions', chunk_predictions)

            file_ids, references, predictions = self.convert_chunks_to_files(val_dataset, chunk_predictions)
            m = self._metrics(predictions, references)
            print('m', m)

            # input('waiting...')

            return {'eval_accuracy': m['accuracy']}

        class StepCallback(TrainerCallback):

            def on_log(self, args, state, control, **kwargs):
                # if state.global_step % 5 == 0:
                print('\nCalling StepCallback.on_log()')
                msg = 'MalGPT - {}%\n'.format(round(100*state.global_step/state.max_steps, 2))
                msg += '---------------\n'
                msg += '{} epochs of {}\n'.format(round(state.epoch, 2), state.num_train_epochs)
                msg += '{} steps of {}\n'.format(state.global_step, state.max_steps)
                whats.send(msg)
                # print('Exiting StepCallback.on_log()')

            def on_train_end(self, args, state, control, **kwargs):
                print('\nCalling StepCallback.on_train_end()')
                print(state)


        class MyEarlyStoppingCallback(EarlyStoppingCallback):
            def on_log(self, args, state, control, **kwargs):
                print('\nCalling MyEarlyStoppingCallback.on_log()')
                print('state', state)
                print(dir(state))
                print('Exiting MyEarlyStoppingCallback.on_log()')

        if self.early_stop:
            early_stop_compute_metrics = compute_metrics
            early_stop_callbacks = [
                StepCallback(),
                EarlyStoppingCallback(early_stopping_patience=3)
            ]
        else:
            early_stop_compute_metrics = None
            early_stop_callbacks = []

        trainer = Trainer(
            model = model, 
            args = training_args,
            train_dataset = train_dataset,
            eval_dataset = val_dataset, 
            data_collator = self._data_collator,

            compute_metrics = early_stop_compute_metrics,
            callbacks = early_stop_callbacks
        )

        try:
          train = trainer.train(resume_from_checkpoint = True)
        except:
          train = trainer.train()
        
        print('Model: saving trained model...')
        trainer.save_model(self.path_model_trained)
        print('Model: model saved!')

        with open(self.path_train_metrics, 'w+') as f:
            f.write(str(train.metrics))

        return train.metrics

    def results(self):
        rv = []
        with open(self.path_results, 'r') as f:
            for line in f.readlines():
                items = line.split(',')
                rv.append([ items[0], int(items[1]) ])
        return rv
    
    def _metrics(self, predictions, references):
        print('gpt._metrics')

        good = 0
        bad = 0
        for pred, ref in zip(references, predictions):
            if pred == ref: good += 1
            else: bad += 1

        print(' - GOOD:', good)
        print(' - BAD:', bad)

        msg = 'MalGPT - Test metrics\n'
        msg += '---------------\n'
        msg += '{} epochs\n'.format(self.epochs)
        msg += 'Acc {}\n'.format(round(100*good/(bad+good), 4))
        msg += 'Good {}\n'.format(good)
        msg += 'Bad {}\n'.format(bad)
        whats.send(msg)

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

    def metrics(self):
        '''
            - read results from file
            - calculate metrics
        '''
        print('gpt.metrics > self.path.results', self.path_results)
        predictions = []
        references = []
        with open(self.path_results, 'r') as f:
            for line in list(f.readlines())[1:]:
                items = line.split(',')
                pred = int(items[1].strip())
                ref = int(items[2].strip())
                predictions.append(pred)
                references.append(ref)                      

        cm = confusion_matrix(references, predictions)
        print(cm)

        m = self._metrics(references, predictions) 
        return m

    def convert_chunks_to_files(self, chunk_dataset, chunk_predictions):
        '''
            Convert test by chunk to test by file
            references are infered from dataset
        '''

        # metrics by file
        files = {}
        for i in range(len(chunk_dataset)):
            file_id, _, _, label = chunk_dataset[i]
            # print('file_id', file_id, 'label', int(label))

            pred = chunk_predictions[i]
            # print('pred', pred)

            if file_id not in files:
                files.update({ file_id:{'pred': [], 'ref': []} })

            files[file_id]['pred'].append(pred)
            files[file_id]['ref'].append(int(label))

        file_ids = []
        references = []
        predictions = []

        for file_id, values in files.items():
            file_ids.append(file_id)
            ref = values['ref'][0]

            # counter
            pred_3c = Counter(values['pred'])
            # print('pred_3c', pred_3c)
            pred_3 = pred_3c.most_common()[0][0]

            references.append(ref)
            predictions.append(pred_3)

        return file_ids, references, predictions


    def test(self, test_dataset):
        
        model_trained = self._get_model_trained()
        trainer = Trainer(
            model = model_trained,
            data_collator = self._data_collator
        )

        # metrics by chunk
        chunk_predictions_weights = trainer.predict(test_dataset)
        print('chunk_predictions_weights len', len(chunk_predictions_weights[0]))
        print('test metrics', chunk_predictions_weights[2])

        chunk_predictions = np.argmax(chunk_predictions_weights[0], axis=1)
        print('chunk_predictions len', len(chunk_predictions))
        

        file_ids, references, predictions = self.convert_chunks_to_files(test_dataset, chunk_predictions)

        print('Final')
        print(' - file_ids', len(file_ids))
        print(' - references', len(references))
        print(' - predictions', len(predictions))

        # save to file
        with open(self.path_results, 'w+') as f:
            f.write(f'file_id,pred_{self.model_label},ref\n')
            for file_id, ref, pred in zip(file_ids, references, predictions):
                f.write(f'{file_id},{pred},{ref}\n')

        # print('CHUNK SCORE')
        # chunk_references = chunk_predictions_weights[1]
        # self._metrics(chunk_predictions, chunk_references)

        # print('FILE SCORE')
        m = self._metrics(predictions, references)
        return m















