import settings

from processor import Processor
from gpt import GPT

# proc = Processor(
#     docs_limit = 10000, # settings.DOCS_LIMIT,
#     chars_to_remove = settings.NPL_CHARS_TO_REMOVE,
#     words_to_remove = settings.NPL_WORDS_TO_REMOVE,
#     path_data_raw = settings.PATH_DATA_ASM,
#     path_data_proc = settings.PATH_DATA_PROC_1,
#     vocab = settings.NPL_VOCAB,
#     max_workers = 2 # settings.MAX_WORKERS
# )
# proc.process_all_docs()

# GTP
gpt_ = GPT()

#prompt = 'The key benefits of TF-IDF are'
prompt = '''
mov ebp var_A dx
mov edx ebp var_E
mov'''
# mov eax ebp arg_4
# movzx ecx word ptr eax edx 2
# push ecx
#'''
completion = gpt_.complete(prompt)



