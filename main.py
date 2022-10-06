import settings

from processor import Processor
from gpt import GPT


proc = Processor()
#proc.process_all_docs()
#proc.split_by_label()

#prompt = '<|startoftext|>'
prompt = 'Four women'
#prompt = 'The key benefits of TF-IDF are'
#prompt = 'mov ebp var_A dx mov edx ebp var_E'

model_name = 'gpt2'

# GTP
gpt_original = GPT(model_name=model_name, for_training=False)
completion = gpt_original.complete(prompt)

print('-----------------')
print(prompt)
print('-----------------')
print(completion)
print('-----------------')

# GTP to train
gpt_train = GPT(model_name=model_name, for_training=True)
gpt_train.train()
completion = gpt_train.complete(prompt)

print('-----------------')
print(prompt)
print('-----------------')
print(completion)
print('-----------------')