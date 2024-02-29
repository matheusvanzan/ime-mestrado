# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import collections
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from keras.utils import np_utils 
from keras.models import Model, Sequential
from keras import layers

from mem import check_mem


class NNetworkManager:

    def __init__(self, path_csv, len_classes, batch_size, epochs):

        self.path_csv = path_csv
        self.len_classes = len_classes
        self.batch_size = batch_size
        self.epochs = epochs
        # self.features = features

        self.model = None

    def print_train_test_split(self, X_train, X_test, Y_train, Y_test):
        print('X_train', X_train.shape, type(X_train))
        print(X_train)
        print('X_test', X_test.shape, type(X_test))
        print(X_test)
        # print('Y_train', Y_train.shape, type(Y_train))
        # print(Y_train)
        # print('Y_test', Y_test.shape, type(Y_test))
        # print(Y_test)

    def create_folds(self, n_splits):
        df = pd.read_csv(self.path_csv)
        # df = df[df['Size'] > 0] # != 0

        X = np.array(df['Id']).reshape(-1, 1) # matriz coluna
        y = np.array(df['Class'])

        skf = StratifiedKFold(
            n_splits = n_splits,
            shuffle = True,
            random_state = 42
        )

        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            # print(i, "TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            Y_train = np_utils.to_categorical(y_train, self.len_classes) # one-hot format
            Y_test = np_utils.to_categorical(y_test, self.len_classes)

            yield (i, X_train, X_test, Y_train, Y_test)

    def check_fold_exists(self, path, prefix):
        folds = ['X_train', 'X_test', 'Y_train', 'Y_test']

        return all([
            os.path.exists(f'{path}/{prefix}_{fold}.pkl') for fold in folds])

    def save_fold(self, path, prefix, X_train, X_test, Y_train, Y_test):
        # pd.DataFrame(df).to_csv(f'{path}/{prefix}_{df_name}.csv', encoding='utf-8')

        self.print_train_test_split(X_train, X_test, Y_train, Y_test)

        with open(f'{path}/{prefix}_X_train.pkl', 'wb') as f:
            pickle.dump(X_train, f)

        with open(f'{path}/{prefix}_X_test.pkl', 'wb') as f:
            pickle.dump(X_test, f)

        with open(f'{path}/{prefix}_Y_train.pkl', 'wb') as f:
            pickle.dump(Y_train, f)

        with open(f'{path}/{prefix}_Y_test.pkl', 'wb') as f:
            pickle.dump(Y_test, f)

    def load_fold(self, path, prefix):
        # df_return.append(pd.read_csv(f'{path}/{prefix}_{df_name}.csv', index_col = 0))
        
        with open(f'{path}/{prefix}_X_train.pkl', 'rb') as f:
            X_train = pickle.load(f)

        with open(f'{path}/{prefix}_X_test.pkl', 'rb') as f:
            X_test = pickle.load(f)

        with open(f'{path}/{prefix}_Y_train.pkl', 'rb') as f:
            Y_train = pickle.load(f)

        with open(f'{path}/{prefix}_Y_test.pkl', 'rb') as f:
            Y_test = pickle.load(f)

        # self.print_train_test_split(X_train, X_test, Y_train, Y_test)

        return (X_train, X_test, Y_train, Y_test)

    def fit(self, path, prefix, X_train, X_test, Y_train, Y_test, cache=False):
        path_model = f'{path}/models/{prefix}_model.pkl'
        # check for cache
        self.model = None
        if cache and os.path.exists(path_model):
            with open(path_model, 'rb') as f:
                self.model = pickle.load(f)
        
        if self.model:
            return self.model
        
        alpha = 0.1

        features_len = int(X_train.shape[1])
        classes_len = int(Y_train.shape[1])
        samples_len = int(X_train.shape[0])
        hidden_dim_len = int( samples_len / (alpha * (features_len + classes_len)) )

        layer_list = [
            layers.Input(shape=(features_len,)),
            layers.Dense(
                units=features_len, 
                activation='relu',
                # use_bias=False
            ),
        ]

        for i in range(1):
            layer_list += [
                layers.Dense(
                    units=5000, 
                    activation='relu',
                    # use_bias=False
                ),
                # layers.Dropout(0.2, noise_shape = None, seed = 42)
            ]

        layer_list += [
            layers.Dense(
                units=classes_len, 
                activation='softmax',
                # use_bias=False
            ),
        ]

        self.model = Sequential(layer_list)

        print(self.model.summary())

        # self.model.add(Dense(features_len, activation='sigmoid')) # relu: most common for hidden layers
        # self.model.add(Dense(int(features_len/2 + classes_len/2), activation='sigmoid'))
        # self.model.add(Dense(512, input_shape=(784,))) #(784,) is not a typo -- that represents a 784 length vector!
        # self.model.add(Activation('relu'))
        # layers.Dropout(rate=0.5, seed=42),
        # Dropout

        # compile network
        self.model.compile(
            loss='categorical_crossentropy', # binary_crossentropy mean_squared_error mse
            optimizer='adam', # adam sgd adadelta
            metrics=[
                'accuracy', # accuracy MeanSquaredError AUC 
            ]
        )

        # fit network
        self.model.fit(X_train, Y_train, 
            epochs = self.epochs, 
            batch_size = self.batch_size,
            verbose=3
        )

        with open(path_model, 'wb') as f:
            pickle.dump(self.model, f)

        # evaluate
        # TODO: evaluate by class
        # loss, acc = model.evaluate(X_train, Y_train, verbose=None)
        # print(f'    - {prefix}: train-acc: {round(100*acc, 2)}%')

        return self.model

    def evaluate(self, X, Y, verbose=None):
        if not self.model:
            raise Exception('model = None. call fit() first')

        return self.model.evaluate(X, Y, verbose=verbose)
    
    def predict(self, X, Y):
        ref_labels = np.argmax(Y, axis=1)
        pred = self.model.predict(X)
        pred_labels = np.argmax(pred, axis=1)
        return pred_labels, ref_labels

    def evaluate_by_class(self, fold_i, X, Y, verbose=None):
        # all
        loss_all, acc_all = self.evaluate(X, Y, verbose=verbose)

        # by class
        df_labels = pd.read_csv(self.path_csv)
        # df_labels.drop(columns='Size', inplace=True)
        df_labels.set_index('Id', inplace=True)
        df_X = pd.merge(X, df_labels, left_index=True, right_index=True)

        fold_list = [fold_i]*(self.len_classes+1)
        i_list = [str(i) for i in range(self.len_classes)]
        loss_list = []
        acc_list = []
        for i in range(self.len_classes):
            df_class = df_X.loc[df_X['Class'] == i]
            X_class = df_class.drop(columns='Class')
            y_class = np.array(df_class['Class'])
            Y_class = np_utils.to_categorical(y_class, self.len_classes)
            loss, acc = self.evaluate(X_class, Y_class, verbose=verbose)
            
            loss_list.append(loss)
            acc_list.append(acc)

        i_list.append('all')
        loss_list.append(loss_all)
        acc_list.append(acc_all)

        # create df
        df = pd.DataFrame([fold_list, i_list, acc_list, loss_list], index=['fold', 'class', 'acc', 'loss']).T
        df.set_index(['fold', 'class'], inplace=True)
        # print(df)
        return df
