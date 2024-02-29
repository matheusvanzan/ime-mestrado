import os

# Debug
# define debug level

# Paths - inside docker
#
# proc-v1 - exclude chars and spaces - only files < 1MB
# proc-v2 - exclude words - all files truncated
# proc-v3 - only vocab - all files
# proc-v4 - only vocab and adjancent hex numbers (digits) >= 3 - all files

PATH_HOME = os.path.expanduser('~')
PATH_DATA = os.path.join(PATH_HOME, 'data')
PATH_DATA_RAW = os.path.join(PATH_DATA, 'raw')
PATH_DATA_COUNTS = os.path.join(PATH_DATA, 'counts')
PATH_DATA_LABELS_CSV = os.path.join(PATH_DATA, 'trainLabels.csv')

PATH_DATA_PROC = os.path.join(PATH_DATA, 'proc-v3')
PATH_DATA_TFIDF = os.path.join(PATH_DATA, 'tfidf-v3')
PATH_DATA_VOCAB = os.path.join(PATH_DATA, 'vocab.csv')

TEST_PATH_DATA = os.path.join(PATH_HOME, 'data', 'test')
TEST_PATH_DATA_RAW = os.path.join(TEST_PATH_DATA, 'raw')
TEST_PATH_DATA_COUNTS = os.path.join(TEST_PATH_DATA, 'counts')

# Parse
DOCS_LIMIT = 100000
MAX_WORKERS = 3

# NNetwork
NN_CLASSES = ['Ramnit','Lollipop','Kelihos_ver3','Vundo','Simda',
                'Tracur','Kelihos_ver1','Obfuscator.ACY','Gatak']
NN_LEN_CLASSES = 9
NN_KFOLD = 10
NN_MEAN_TRIALS = 1

# NPL Tokens
NPL_NGRAM = 1
NPL_MAX_FILESIZE = 0 # in MB - 0 for no limit
NPL_MAX_FEATURES = 1000
NPL_TOKEN_MAX_LEN = 4
NPL_CHARS_TO_REMOVE = ''.join([
    "'",
    '"',
    '.,;:!?', # pontos
    '+-=*<>', # sinais
    '()[]{}', # fechamento
    '\n\r\t', # espacos
    '´^~`', # acentos
    '@#$%¨&|_/'
    '' # \\x19
])
NPL_WORDS_TO_REMOVE = [
    'text',
    's u b r o u t i n e',
    'code xref'
]
NPL_VOCAB = []
with open(PATH_DATA_VOCAB, 'r', encoding='utf-8') as f:
    NPL_VOCAB = set(f.read().split(','))
