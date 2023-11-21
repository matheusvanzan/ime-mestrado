'''

Main code of code

'''

# import settings vars only here, inject them on other instances

import os
import settings
import datetime
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from network import NNetworkManager
from processor import Processor
from npl import NPL



def main(args):

    if args.process:
        processor = Processor(
            docs_limit = settings.DOCS_LIMIT,
            chars_to_remove = settings.NPL_CHARS_TO_REMOVE,
            words_to_remove = settings.NPL_WORDS_TO_REMOVE,
            path_data_raw = settings.PATH_DATA_RAW,
            path_data_proc = settings.PATH_DATA_PROC,
            vocab = settings.NPL_VOCAB,
            max_workers = settings.MAX_WORKERS
        )

        # pre-process files
        # processor.process_all_docs() # agora resp do geral

        return None

   
    # g: ngram
    # ms: max size
    # mf: max features
    g_ms_mf_list = [ 
        (1, 0, 10),
    ]

    type_ = 'tfidf' # 'count'
    path = settings.PATH_DATA_TFIDF # settings.PATH_DATA_COUNTS

    # --------------------
    # CREATE DATASET
    # --------------------

    if args.create:
        for g, ms, mf in g_ms_mf_list:
            network_manager = NNetworkManager(
                path_csv = settings.PATH_DATA_LABELS_CSV
            )
            npl = NPL(
                path_proc = settings.PATH_DATA_PROC,
                max_filesize = ms,
                max_features = mf,
                ngram = g
            )

            folds = network_manager.create_folds(n_splits = settings.NN_KFOLD)
            for i, X_train, X_test, Y_train, Y_test in folds:
                # i, X_train, X_test, Y_train, Y_test = list(folds)[0] # fold 1
                fold_i = i+1
                prefix = f'g{g}_ms{ms}_mf{mf}_f{fold_i}'
                print(prefix)

                # CountVectorizer or TfidfVectorizer
                if network_manager.check_fold_exists(path, prefix):
                    print(f'Skip fold {type_}:{prefix}')
                else:
                    X_train = npl.create_X(X_train, type_)
                    X_test = npl.create_X(X_test, type_)
                    network_manager.save_fold(path, prefix, X_train, X_test, Y_train, Y_test)

    # --------------------
    # TRAIN AND EVALUATE
    # --------------------

    if args.train:
        network_manager = NNetworkManager(
            path_csv = settings.PATH_DATA_LABELS_CSV
        )

        df_folds = []
        folds = network_manager.create_folds(n_splits = settings.NN_KFOLD)
        for i, X_train, X_test, Y_train, Y_test in folds:
            # i, X_train, X_test, Y_train, Y_test = list(folds)[0] # fold 1
            fold_i = i+1

            X_train_list, X_test_list = [], []
            for g, ms, mf in g_ms_mf_list:
                prefix = f'g{g}_ms{ms}_mf{mf}_f{fold_i}'
                # print(prefix)
                X_train, X_test, Y_train, Y_test = network_manager.load_fold(path, prefix)
                X_train_list.append(X_train)
                X_test_list.append(X_test)

            X_train = pd.concat(X_train_list, axis=1)
            X_test  = pd.concat(X_test_list, axis=1)

            prefix_model = '_'.join([f'g{g}_ms{ms}_mf{mf}' for g, ms, mf in g_ms_mf_list]) + f'_f{fold_i}'

            df_trials = []
            for _ in range(settings.NN_MEAN_TRIALS):
                network_manager.fit(path, prefix_model, X_train, X_test, Y_train, Y_test)
                df_trial = network_manager.evaluate_by_class(fold_i, X_test, Y_test)
                df_trials.append(df_trial)

            df_fold = pd.concat(df_trials).groupby(['fold', 'class']).mean()
            df_folds.append(df_fold)
            print(df_fold)

        print('\n\n-- Media dos folds por Classe')
        df_mean = pd.concat(df_folds).groupby(['class']).mean()
        print(df_mean)
        print(str(list(df_mean['acc'])).replace(',', '\n').replace('.', ','))

        print('\n\n-- Media de cada fold')
        df_by_fold = pd.concat([df[df.index.get_level_values('class').isin(['all'])] for df in df_folds])
        print(df_by_fold)
        print(str(list(df_by_fold['acc'])).replace(',', '\n').replace('.', ','))

        mean = df_by_fold.groupby(['class']).mean()
        print(mean)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-p', '--process', dest='process', required=False, action='store_true', help='Process dataset')
    parser.add_argument('-c', '--create', dest='create', required=False, action='store_true', help='Create dataset')
    parser.add_argument('-t', '--train', dest='train', required=False, action='store_true', help='Train the model')
    args = parser.parse_args()
    print(args)

    print(f'pid: {os.getpid()}')
    # input('press any key to continue...')

    start = datetime.datetime.now()
    # pre()
    main(args)
    stop = datetime.datetime.now()
    print(str(stop-start).split('.')[0])