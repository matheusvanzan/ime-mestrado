#!/usr/bin/env python
# coding: utf-8


import os
import torch
import settings

from transformers import GPTJForCausalLM, AutoTokenizer

def gpt_log(prompt, completion):
    pass #path_log
    

class GPT:

    def __init__(self):
        self.device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

        #if self.device_name != 'cuda':
        #    raise Exception('Not using GPU')

        self.device = torch.device(self.device_name)

        self.path_model = os.path.join(settings.PATH_DATA, f'gpt-j.pt')
        self.path_tokenizer = os.path.join(settings.PATH_DATA, f'gpt-j-tok.pt')

        self.model, self.tokenizer = self._get_model_tokenizer()


    def _get_model_tokenizer(self):

        if True: #not os.path.exists(self.path_model):
            #print('Downloading model...')
            model = GPTJForCausalLM.from_pretrained(
                'EleutherAI/gpt-j-6B',
                revision='float16',
                low_cpu_mem_usage=True,
                load_in_8bit=True, 
                device_map='auto'
            )
            #print('Saving model...')
            #torch.save(model, self.path_model)
            #print('Model saved!')
        else:
            print('Using saved model...')
            model = torch.load(self.path_model)

        if True: #not os.path.exists(self.path_tokenizer):
            #print('Downloading tokenizer...')
            #tokenizer = AutoTokenizer.from_pretrained(
            #    'EleutherAI/gpt-j-6B')
            tokenizer = AutoTokenizer.from_pretrained(
                'EleutherAI/gpt-j-6B', 
                bos_token='<|startoftext|>', 
                eos_token='<|endoftext|>', 
                pad_token='<|pad|>')
            #print('Saving tokenizer...')
            #torch.save(tokenizer, self.path_tokenizer)
            #print('Tokenizer saved!')
        else:
            print('Using saved tokenizer...')
            tokenizer = torch.load(self.path_tokenizer)

        return model, tokenizer

    def complete(self, prompt):
        self.model.to(self.device)
        input_ = self.tokenizer(prompt, return_tensors='pt')
        input_.to(self.device)

        gen_tokens = self.model.generate(
            input_.input_ids,
            do_sample=True,
            temperature=0.00001,
            # max_length=100,
            max_new_tokens=10
        )

        self.tokenizer.batch_decode(gen_tokens)
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]

        new_text = gen_text.replace(prompt, '')

        print('---')
        print(prompt)
        print('---')
        print(new_text)

        return new_text




















