#!/usr/bin/env python
# coding: utf-8


import os
from webbrowser import get

import torch
from transformers.models import gpt2
torch.manual_seed(42)

from datetime import datetime
from torch.utils.data import random_split
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPT2LMHeadModel, GPTNeoForCausalLM

import pandas as pd
import numpy as np

import settings
from dataset import DirectoryDataset

from sentence_transformers import SentenceTransformer, util

from datasets import load_metric
metric = load_metric('accuracy')


class GPT:

    def __init__(self, model_name, model_label):
        print('Model: name', model_name)
        print('Model: label', model_label)

        self.limit = settings.DATASET_LIMIT

        self.chunk_size = settings.GPT_TRAIN_CHUNK_SIZE
        self.batch_size = settings.GPT_BATCH_SIZE
        self.epochs = settings.GPT_EPOCHS

        self.model_name = model_name
        self.model_label = model_label

        path = os.path.join(settings.PATH_DATA, model_name)
        self.path_tokenizer = os.path.join(path, 'tokenizer')
        self.path_model = os.path.join(path, 'model')

        # Ex: 5.limit-256.chunk-16.epochs-2.batch-16
        path_label_name = '{}.limit-{}.chunk-{}.epochs-{}.batch-{}'.format(
            self.model_label,
            self.limit,
            self.chunk_size,
            self.epochs,
            self.batch_size
        )
        path_label = os.path.join(path, path_label_name)
        self.output_dir = os.path.join(path_label, 'partial')
        self.logging_dir = os.path.join(path_label, 'logs')
        self.path_model_trained = os.path.join(path_label, 'model-trained')

        self._create_paths()

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
                bos_token='<|startoftext|>',
                eos_token='<|endoftext|>', 
                pad_token='<|pad|>'
            )
            print('Model: saving tokenizer...')
            torch.save(tokenizer, self.path_tokenizer)
            print('Model: tokenizer saved!')
        else:
            print('Model: using saved tokenizer...')
            tokenizer = torch.load(self.path_tokenizer)

        return tokenizer

    def _get_model_original(self):

        if not os.path.exists(self.path_model):
            print('Model: downloading model...')
            model = GPT2LMHeadModel.from_pretrained(
                self.model_name
            ) #.cuda()
            # model.resize_token_embeddings(len(self.tokenizer))
            print('Model: saving model...')
            torch.save(model, self.path_model)
            print('Model: model saved!')
        else:
            print('Model: using saved model...')
            model = torch.load(self.path_model)

        model.to(torch.device('cuda'))

        return model

    def _get_model_trained(self):

        print(f'Model: using trained at {self.path_model_trained} ...')
        model = GPT2LMHeadModel.from_pretrained(
            self.path_model_trained
        )
        model.to(torch.device('cuda'))
        return model

    def train(self, train_dataset, val_dataset):

        model = self._get_model_original()

        training_args = TrainingArguments(
            output_dir = self.output_dir,
            logging_dir = self.logging_dir,

            evaluation_strategy='steps',
            eval_steps=100,

            logging_strategy='steps',
            logging_steps=100,

            save_strategy='steps',
            save_steps=1000,

            num_train_epochs = self.epochs,
                                           
            per_device_train_batch_size = self.batch_size,
            per_device_eval_batch_size = self.batch_size,

            warmup_steps=0,
            weight_decay=0.01,

            # load_best_model_at_end=True,
            # metric_for_best_model='accuracy',
            report_to='tensorboard'
        )

        # def compute_metrics(eval_pred):
        #     print('\n\ncompute_metrics\n\n', 'eval_pred', eval_pred)
        #     # predictions, labels = eval_pred
        #     # predictions = np.argmax(predictions, axis=1)
        #     # results =  metric.compute(predictions=predictions, references=labels)
        #     # return results
        #     return {'accuracy': 0.75}

        trainer = Trainer(
            model=model, 
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset, 
            # This custom collate function is necessary 
            # to built batches of data
            data_collator=lambda data: {
                'input_ids': torch.stack([f[0] for f in data]),       
                'attention_mask': torch.stack([f[1] for f in data]),
                'labels': torch.stack([f[0] for f in data])
            },
            # compute_metrics=compute_metrics,
        )
        # Start training process!
        #trainer.add_callback(CustomCallback(trainer))

        train = trainer.train() # trainer.train("checkpoint-9500")

        # print(train.metrics)

        print('Model: saving trained model...')
        trainer.save_model(self.path_model_trained)
        print('Model: model saved!')

    def encode(self, prompt):
        gen = self.tokenizer(prompt, return_tensors='pt')
        return gen.input_ids, gen.attention_mask

    def decode(self, input_ids):
        return self.tokenizer.decode(input_ids, skip_special_tokens=True).strip()


    def complete(self, prompt):
        '''
            input: str
            output: str
        '''

        input_ids, attention_mask = self.encode(prompt)

        sample_output = self.complete_ids(input_ids[0], attention_mask[0])
        completion = self.decode(sample_output)
        return completion.strip()

    def complete_ids(self, input_ids, attention_mask):
        '''
            input:
                input_ids, attention_mask
            output:
                ids
        '''

        if not self.model:
            self.model = self._get_model_trained()

        # print('  input_ids', input_ids)
        len_input_ids = len(input_ids)

        input_ids = torch.tensor([list(input_ids)])
        attention_mask = torch.tensor([list(attention_mask)])
        
        # Generate 
        sample_outputs = self.model.generate(
            
            inputs = input_ids.cuda(), 
            attention_mask = attention_mask.cuda(),

            pad_token_id = self.pad_token_id,
            max_new_tokens = self.chunk_size,

            # to get a more deterministic completion
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.0001,              
            num_return_sequences=1
        )

        # TODO: handle 0-size tensor (num_return_sequences)
        output_ids = sample_outputs[0][len_input_ids:]
        # print('  output_ids', output_ids)
        return output_ids

    def complete2(self, prompt):

        input_ids, attention_mask = self.encode(prompt)

        if not self.model:
            self.model = self._get_model_trained()
        
        # Generate 
        sample_outputs = self.model.generate(
            
            inputs = input_ids.cuda(), 
            attention_mask = attention_mask.cuda(),

            pad_token_id = self.pad_token_id,
            max_new_tokens = self.chunk_size,

            # to get a more deterministic completion
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.0001,              
            num_return_sequences=1
        )

        # TODO: handle 0-size tensor (num_return_sequences)
        output_ids = sample_outputs[0]
        # print('  output_ids', output_ids)
        completion = self.decode(output_ids)
        return completion.strip()


class GPTTester:

    def __init__(self, gpt_list, dataset_list):
        self.gpt_list = gpt_list
        self.dataset_list = dataset_list # only test_dataset

        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def cos_similarity(self, s1, s2):
        # print('s1', s1)
        # print('s2', s2)

        e1 = self.sentence_model.encode(s1, convert_to_tensor=True)
        e2 = self.sentence_model.encode(s2, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(e1, e2)

        # print('cos_similarity', sim[0][0])
        return float(sim[0][0])

    def similarity(self, s1, s2):
        '''
            returns: float number in (0, 1)
        '''
        # TODO: test another forms
        return self.cos_similarity(s1, s2)        

    def check(self):
        # prompt = 'var epb mov 100000h'
        prompt = 'assume cs 10001000 assume es nothing ss nothing ds data fs'
        sample_output = 'nothing gs nothing pusha upx1 pusha upx1 pusha'

        print('-----------')
        print('prompt:', prompt)
        print('sample_output:', sample_output)
        print('-----------')

        for gpt in self.gpt_list:
            print('GPT:', gpt.model_label)
            output = gpt.complete(prompt)
            print('  output:', output)
            print('  similarity:', self.similarity(output, sample_output))

    def test(self):

        df_data = []
        df_true = 0

        # FOR EACH DATASET
        for test_dataset in self.dataset_list:
            dataset_id = test_dataset.path.split('/')[-1]
            # print('Dataset', dataset_id)        

            test_dataset_list = list(test_dataset) # puts generator into memmory
            len_dataset_list = len(test_dataset_list)

            # FOR EACH FILE/CHUNK
            percent_last = 0
            start = datetime.now()
            for i in range(0, len_dataset_list - 1):

                ### print log
                percent = int(100*i/len_dataset_list)
                if percent != percent_last and percent % 5 == 0:
                    percent_last = percent
                    now = datetime.now()
                    elapsed = now - start
                    remain = (elapsed/percent)*(100-percent)
                    to_print = f"Test: dataset {dataset_id} - {percent}% "
                    to_print += f"({str(elapsed).split('.')[0]} < {str(remain).split('.')[0]})"
                    print(to_print)
                ### end print log

                input_ids, attention_mask = test_dataset_list[i]
                input_ids_next, _ = test_dataset_list[i+1]

                gpt0 = self.gpt_list[0]
                # prompt = gpt0.decode(input_ids)
                prompt_next = gpt0.decode(input_ids_next) # da pra economizar aqui? , para a proxima iteração, criar um array de decodes (salvar encoded e decoded???)

                sim_list = []
                sim_max = 0
                sim_max_id = 0

                for gpt in self.gpt_list:
                    # output = gpt.complete(prompt)
                    output_ids = gpt.complete_ids(input_ids, attention_mask)
                    output = gpt.decode(output_ids)
                    # print('  output:', output)

                    sim = self.similarity(
                        output, # prediction
                        prompt_next # actual value
                    )
                    sim_list.append(sim)

                    if sim > sim_max:
                        sim_max = sim
                        sim_max_id = gpt.model_label

                if str(dataset_id) == str(sim_max_id):
                    df_true += 1

                df_data.append([dataset_id, i, sim_max_id] + sim_list)

        columns = ['dataset', 'chunk', 'sim_max'] + \
                  [f'sim_{gpt.model_label}' for gpt in self.gpt_list]

        df = pd.DataFrame(df_data, columns=columns)
        acc = df_true/len(df_data)
        return df, acc


        # metric = load_metric("accuracy")
        # metric.compute(predictions=[0,0,1,1], references=[0,1,1,1])
        # # {'accuracy': 0.75}
            


class GPTOriginal(GPT):

    def __init__(self, model_name):
        return super().__init__(model_name, 'original')

    def _get_tokenizer(self):

        if not os.path.exists(self.path_tokenizer):
            print('Model: downloading tokenizer...')
            tokenizer = GPT2Tokenizer.from_pretrained(
                self.model_name
            )
            print('Model: saving tokenizer...')
            torch.save(tokenizer, self.path_tokenizer)
            print('Model: tokenizer saved!')
        else:
            print('Model: using saved tokenizer...')
            tokenizer = torch.load(self.path_tokenizer)

        return tokenizer

    def complete(self, prompt):

        model = self._get_model_original()
        generated = self.tokenizer(prompt, return_tensors='pt').input_ids.cuda()

        # Generate 3 movie descriptions
        sample_outputs = model.generate(
            generated, 
            # Use sampling instead of greedy decoding 
            do_sample=True, 
            # Keep only top 50 token with the highest probability
            top_k=50, 
            # Maximum sequence length
            # max_length=50,
            max_new_tokens=30,
            # Keep only the most probable tokens with cumulative probability of 95%
            top_p=0.95, 
            # Changes randomness of generated sequences
            temperature=0.0001,
            # Number of sequences to generate                 
            num_return_sequences=1
        )

        # Print generated descriptions
        # for i, sample_output in enumerate(sample_outputs): 
        #     print('{}: {}'.format(i, self.tokenizer.decode(sample_output, skip_special_tokens=True)).replace('\n', ' '))

        completion = self.tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
        return completion[len(prompt):].replace('\n', ' ').strip()

