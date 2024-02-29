import pickle
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-p', '--path', dest='path', \
    required=False, help='')
args = parser.parse_args()


with open(args.path, 'rb') as f:
    vocab = pickle.load(f)

for i, (key, value) in enumerate(vocab.items()):
    print(i, key, value)