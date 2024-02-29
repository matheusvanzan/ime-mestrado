# Extract words from .asm files and put it into .csv file

#POC - convert Count Vectorizer to TDIDF Vectorizer

import os
import time
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


DATA_RAW_PATH = '/home/docky/data/train/raw/'
DATA_BOW_PATH = '/home/docky/data/train/bow/'
CHARS_TO_REMOVE = [',', ';', ':', '.']

MAX_FEATURES = 5 # None

def count_to_tfidf(count_df):
    transformer = TfidfTransformer()
    tfidf_trans = transformer.fit_transform(count_df)
    tfidf_vect = pd.DataFrame(
        tfidf_trans.toarray(), 
        index = count_df.index, 
        columns = count_df.columns
    )
    return tfidf_vect


def main():

    labels = []
    docs = []
    corpus = []

    i = 0
    for filename in os.listdir(DATA_RAW_PATH):
        i = i+1
        docs.append(filename.split('.')[0])
        print(i, filename)
        filepath = os.path.join(DATA_RAW_PATH, filename)

        with open(filepath, encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        for char_to_remove in CHARS_TO_REMOVE:
            content.replace(char_to_remove, ' ')
        corpus.append(content)

    count_vect = CountVectorizer(max_features=MAX_FEATURES)
    tfidf_vect = TfidfVectorizer(max_features=MAX_FEATURES)
    # vectorizer = TfidfVectorizer()
    #     stop_words='english', 
    #     analyzer='word', # 'char_wb'
    #     max_features=MAX_FEATURES)

    count_X = count_vect.fit_transform(corpus)
    tfidf_X = tfidf_vect.fit_transform(corpus)

    count_features = count_vect.get_feature_names_out()
    tfidf_features = tfidf_vect.get_feature_names_out()

    count_df = pd.DataFrame(
        data = count_X.toarray(),
        index = docs,
        columns = count_features
    )
    tfidf_df = pd.DataFrame(
        data = tfidf_X.toarray(),
        index = docs,
        columns = tfidf_features
    )

    print('Count Vectorizer\n')
    print(count_df)

    print('TD-IDF Vectorizer\n')
    print(tfidf_df)

    conv_tfidf_df = count_to_tfidf(count_df)
    print('Converted TD-IDF Vectorizer\n')
    print(conv_tfidf_df)
        

if __name__ == '__main__':
    start = time.time()
    main()
    stop = time.time()
    print(round(stop-start))