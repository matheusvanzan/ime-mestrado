#!/usr/bin/env python
# coding: utf-8


import os
from webbrowser import get
import torch

from torch.utils.data import random_split
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPT2LMHeadModel, GPTNeoForCausalLM

import settings
from dataset import get_dataset


# Set the random seed to a fixed value to get reproducible results 
torch.manual_seed(42)


class GPT:

    def __init__(self, model_name, for_training=True):

        print('=================')
        print('Using model:', model_name)
        print('Will train?', for_training)
        print('=================')

        self.model_name = model_name

        path = os.path.join(settings.PATH_DATA, model_name)

        self.output_dir = os.path.join(path, 'partial')
        self.logging_dir = os.path.join(path, 'logs')
        
        self.path_tokenizer = os.path.join(path, 'tokenizer')
        self.path_tokenizer_trained = os.path.join(path, 'tokenizer-trained')

        if for_training:
            self.path_model = os.path.join(path, 'model-trained')

            self.tokenizer = self._get_tokenizer_for_training()
            self.model = self._get_model()
        else:
            self.path_model = os.path.join(path, 'model')

            self.tokenizer = self._get_tokenizer()
            self.model = self._get_model()


    def _get_tokenizer(self):

        if not os.path.exists(self.path_tokenizer):
            print('Downloading tokenizer...')
            tokenizer = GPT2Tokenizer.from_pretrained(
                self.model_name
            )
            print('Saving tokenizer...')
            torch.save(tokenizer, self.path_tokenizer)
            print('Tokenizer saved!')
        else:
            print('Using saved tokenizer...')
            tokenizer = torch.load(self.path_tokenizer)

        return tokenizer

    def _get_tokenizer_for_training(self):

        if not os.path.exists(self.path_tokenizer_trained):
            print('Downloading tokenizer...')  
            tokenizer = GPT2Tokenizer.from_pretrained(
                self.model_name, 
                bos_token='<|startoftext|>',
                eos_token='<|endoftext|>', 
                pad_token='<|pad|>'
            )
            print('Saving tokenizer...')
            torch.save(tokenizer, self.path_tokenizer_trained)
            print('Tokenizer saved!')
        else:
            print('Using saved tokenizer...')
            tokenizer = torch.load(self.path_tokenizer_trained)

        return tokenizer

    def _get_model(self):

        if not os.path.exists(self.path_model):
            print('Downloading model...')
            model = GPT2LMHeadModel.from_pretrained(
                self.model_name
            ) #.cuda()
            model.resize_token_embeddings(len(self.tokenizer))
            print('Saving model...')
            torch.save(model, self.path_model)
            print('Model saved!')
        else:
            print('Using saved model...')
            model = torch.load(self.path_model)

        model.to(torch.device('cuda'))

        return model
    

    def train(self):

        dataset = get_dataset(self.tokenizer)
        print('len(dataset)', len(dataset))

        train_size = int(0.8 * len(dataset))
        train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

        training_args = TrainingArguments(
            output_dir = self.output_dir,
            logging_dir = self.logging_dir,

            num_train_epochs=5,
            logging_steps=5000,
            save_steps=5000,                                   
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            warmup_steps=100,
            weight_decay=0.01
        )

        trainer = Trainer(
            model=self.model, 
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
            #compute_metrics=compute_metrics,
        )
        # Start training process!
        #trainer.add_callback(CustomCallback(trainer)) 
        train = trainer.train()

        print(train.metrics)

        # trainer.save_model(os.path.join(PATH_DATA, model_name, 'model-trained'))

        

    def complete(self, prompt):
        generated = self.tokenizer(prompt, return_tensors='pt').input_ids.cuda()

        # Generate 3 movie descriptions
        sample_outputs = self.model.generate(generated, 
            # Use sampling instead of greedy decoding 
            do_sample=True, 
            # Keep only top 50 token with the highest probability
            top_k=50, 
            # Maximum sequence length
            max_length=50,
            #max_new_tokens=100,
            # Keep only the most probable tokens with cumulative probability of 95%
            top_p=0.95, 
            # Changes randomness of generated sequences to 1.9
            temperature=0.0001,
            # Number of sequences to generate                 
            num_return_sequences=3
        )

        # Print generated descriptions
        for i, sample_output in enumerate(sample_outputs): 
            print('{}: {}'.format(i, self.tokenizer.decode(sample_output, skip_special_tokens=True)).replace('\n', ' '))

        completion = self.tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
        return completion[len(prompt):].replace('\n', ' ')




