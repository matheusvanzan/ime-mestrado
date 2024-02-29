import os
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from mem import check_mem


class NPL:

    def __init__(self, path_proc, max_filesize, max_features, ngram):
        self.path_proc = os.path.join(path_proc, 'all')
        self.max_filesize = max_filesize
        self.max_features = max_features
        self.ngram = ngram

    def build_tokenizer(self, content, ngram=1, token_min_len=2):
        
        tokens = []
        words = content.split(' ')
        for i in range(len(words) - ngram + 1):
            token_words = \
                [words[i+j] for j in range(ngram) \
                    if len(words[i+j]) >= token_min_len]
            
            if len(token_words) == ngram and \
                not all([w.isdigit() for w in token_words]):
                tokens.append(' '.join(token_words))

        return tokens

    def create_corpus(self, X):
        corpus = []
        docs = [x[0] for x in X]
        values = []
        for i, doc in enumerate(docs):
            values.append((i, X.shape[0], doc))
        
        with ProcessPoolExecutor(max_workers = 6) as executor:
            for i, content in enumerate(executor.map( \
                self.create_doc_corpus, values)):
                corpus.append(content)
            
        return corpus

    def create_doc_corpus(self, value):
        i, total, doc = value
        path_doc = os.path.join(self.path_proc, f'{doc}.asm')

        print(f'{i} of {total} - {doc}')
        check_mem()
        with open(path_doc, encoding='utf-8', errors='ignore') as f:
            content = f.read()

        len_ = len(content)
        if self.max_filesize != 0:
            max_filesize_mb = int(self.max_filesize*1024*1024)
            len_ = min(max_filesize_mb, len_) # limit in MB
            
        content = content[:len_]
        return content

    def create_X(self, X, type_):
        corpus = self.create_corpus(X)       

        # build vect
        start = datetime.now()
        docs = [x[0] for x in X]

        # def tokenizer(content):
        #     return self.build_tokenizer(content, ngram=self.ngram)

        if type_ == 'count':
            print('CountVectorizer ...')
            vect = CountVectorizer(
                max_features = self.max_features,
                ngram_range = (self.ngram, self.ngram)
                # tokenizer = tokenizer
            )

        elif type_ == 'tfidf':
            print('TfidfVectorizer ...')
            vect = TfidfVectorizer(
                max_features = self.max_features,
                ngram_range = (self.ngram, self.ngram),
                # tokenizer = tokenizer,
                # vocabulary = ['loc', 'mov', 'push', 'sub', 'text']
            )

        vect_X = vect.fit_transform(corpus)
        features = vect.get_feature_names_out()
        df = pd.DataFrame(
            data = vect_X.toarray(),
            index = docs,
            columns = features
        )
        stop = datetime.now()
        print(stop-start)

        return df