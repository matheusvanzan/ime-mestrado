# Extract words from .asm files and put it into .csv file

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import pandas as pd
import numpy as np

from itertools import combinations

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split

from keras.utils import np_utils 
from keras.models import Sequential
from keras.layers import Dense

from utils import add_dfs
from tokenizer import TokenizerManager


# NPL params

MAX_FEATURES =  50 # None


TOKEN_WINDOW = 3
TOKEN_MIN_LEN = 1

# NN params
LEN_CLASSES = 9 # 1 ... 9
EPOCHS = 100

def count_to_tfidf(count_df):
    transformer = TfidfTransformer()
    tfidf_trans = transformer.fit_transform(count_df)
    tfidf_vect = pd.DataFrame(
        tfidf_trans.toarray(), 
        index = count_df.index, 
        columns = count_df.columns
    )
    return tfidf_vect

    #
    # remember not to remove repeated tokens
    # they set the count and frequency in the Vectorizer
    #
    tokens = []
    words = content.split(' ')
    for i in range(len(words) - window + 1):
        token_words = \
            [words[i+j] for j in range(window) \
                if len(words[i+j]) >= token_min_len]
        
        if len(token_words) == window:
            tokens.append(' '.join(token_words))

    print(len(tokens), 'tokens', tokens[:5], '...')
    return tokens

def count_df_from_docs(filepaths, filetype='.asm'):

    labels = []
    docs = []
    corpus = []

    i = 0
    for filepath in filepaths:

        type_ = os.path.splitext(filepath)[-1]
        if type_ != filetype:
            break

        i = i+1
        doc = os.path.basename(filepath)
        doc = doc.replace(filetype, '')
        docs.append(doc)
        # print(i, 'of', len(filepaths), doc)

        with open(filepath, encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        for char_to_remove in CHARS_TO_REMOVE:
            content = content.replace(char_to_remove, ' ')

        content = ' '.join(content.split()) # remove multiple whitespaces
        corpus.append(content)

    def tokenizer(content):
        return Tokenizer(content).build_tokenizer( \
            window=TOKEN_WINDOW, token_min_len=TOKEN_MIN_LEN)

    count_vect = CountVectorizer(
        max_features = MAX_FEATURES,
        tokenizer = tokenizer
    )
    count_X = count_vect.fit_transform(corpus)
    count_features = count_vect.get_feature_names_out()
    count_df = pd.DataFrame(
        data = count_X.toarray(),
        index = docs,
        columns = count_features
    )

    return count_df


def main():

    # TODO: create packages with dependency injection
    # models (list of layers)
    # vectorizers (count, tfidf)

    # TODO: tests
    # create mockups for DI classes

    # create CountVectorizer from docs
    labels = []
    paths = []
    corpus = []

    i = 0
    for filename in os.listdir(DATA_RAW_PATH):
        if i < DOCS_LIMIT:
            i = i+1
            filepath = os.path.join(DATA_RAW_PATH, filename)
            paths.append(filepath)

    df_features = count_df_from_docs(paths)
    # print(df_features)

    # read labels for docs
    df_labels = pd.read_csv(os.path.join(DATA_RAW_PATH, 'trainLabels_0.csv'))
    df_labels.set_index('Id', inplace=True)
    # print(df_labels)
    # df_labels_sample = df_labels.loc[df_labels.index.isin(df_features.index)]
    # print(df_labels_sample)

    # by default acts like inner join, dont need sample isin()
    df = pd.merge(df_features, df_labels, left_index=True, right_index=True)
    print(df)

    # TODO: (optional) save new df to .csv file

    # get X, y, train and test
    features = list(df.columns).copy()
    features.remove('Class')
    X = np.array(df[features].values) # .reshape(6, 10) # lembrar oq reshape faz
    y = np.array([df['Class'].values]).T # .reshape(-1, 1)

    # modify Class to be in the one-hot format
    # 0 -> [1, 0, 0, 0, 0, 0, 0, 0, 0]
    # 1 -> [0, 1, 0, 0, 0, 0, 0, 0, 0]
    # 2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0]

    Y = np_utils.to_categorical(y, LEN_CLASSES)

    print('X.shape', X.shape)
    print('Y.shape', Y.shape)

    #
    # training
    #
    
    # random? each training will have diff results
    # split docs before??
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, 
        test_size = 0.2, 
        random_state = 42 # keeps same split always (maybe split docs before)
    )

    # define network
    model = Sequential()
    model.add(Dense(len(features), activation='sigmoid'))
    model.add(Dense(LEN_CLASSES, activation='sigmoid'))

    # notes for future
    # model.add(Dense(512, input_shape=(784,))) #(784,) is not a typo -- that represents a 784 length vector!
    # model.add(Activation('relu'))

    # compile network
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # fit network
    model.fit(X_train, Y_train, epochs=EPOCHS, verbose=None)

    # evaluate
    # TODO: evaluate by class
    loss, acc = model.evaluate(X_test, Y_test, verbose=None)
    print(f'Test result - Accuracy: {100*acc}, Loss {loss}')
        

if __name__ == '__main__':
    start = time.time()
    main()
    stop = time.time()
    print(round((stop-start)/60), 'min')