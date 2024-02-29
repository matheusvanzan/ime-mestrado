'''
Projeto de PAA

Analisar o gasto em tempo e memoria de um processo

Classificacao de caracteres utilizando redes neurais
'''

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import urllib.request
import numpy
import random
import time
import pickle
import numpy as np

from argparse import ArgumentParser

from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras import layers

def reset_random_seeds():
    seed_value = 42
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

def main(optmizer, epochs, layer_dim, quant_layers):
    # we need to tell numpy the dimensions of our arrays
    X = numpy.empty([0, 256])
    y = numpy.empty([0, 10])

    path = '/home/docky/data/semeion.txt'

    with open(path, 'r') as f:
        for line in f.readlines():
            numbers = line.split(' ')
            sample = numpy.array([ float(x) for x in numbers[0:256] ])
            result = numpy.array([ int(x) for x in numbers[256:266] ])

            # after that, append freshly read sample and result to arrays
            X = numpy.concatenate( (X, numpy.array([sample])), axis=0)
            y = numpy.concatenate((y, numpy.array([result])), axis=0)

    # print('X', X.shape)
    # print('y', y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # print('X_train', X_train.shape, 'X_test', X_test.shape)
    # print('y_train', y_train.shape, 'y_test', y_test.shape)

    # define network
    layers_list = [
        layers.Input(shape=(256,)), # has to be 256 (image size)
        layers.Dense(units=256, activation='relu')
    ]

    for _ in range(quant_layers):
        layers_list.append(
            layers.Dense(units=layer_dim, activation='relu')
        )

    layers_list.append(
        layers.Dense(units=10, activation='softmax') # has to be 10 (class size)
    )

    model = Sequential(layers_list)

    # compile network
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=optmizer, 
        metrics=['accuracy']
    )

    start = time.time()
    pstart = time.process_time()

    # fit network
    model.fit(X_train, y_train, epochs=epochs, verbose=None)

    stop = time.time()
    pstop = time.process_time()
    time_elapsed = stop - start
    ptime_elapsed = pstop - pstart

    params = model.count_params()

    # evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=None)

    # stop = time.time()
    # time_elapsed = stop-start
    # print('stop_2', stop - stop_1)

    return params, acc, time_elapsed, ptime_elapsed


if __name__ == '__main__':

    # gc.collect()
    # print(gc.get_stats())

    reset_random_seeds()

    parser = ArgumentParser()
    parser.add_argument('-d', '--dest', dest='dest', help='')
    parser.add_argument('-o', '--optmizer', dest='optmizer', type=str)
    parser.add_argument('-e', '--epochs', dest='epochs', type=int)
    parser.add_argument('-l', '--layer_dim', dest='layer_dim', type=int)
    parser.add_argument('-n', '--n_layers', dest='n_layers', type=int)
    args = parser.parse_args()
    # print(args)

    pid = os.getpid()
    # print(pid)

    params, acc, time_elapsed, ptime_elapsed = main(
        args.optmizer, 
        args.epochs, 
        args.layer_dim, 
        args.n_layers
    )

    return_dict = {
        'pid': pid,
        'optmizer': args.optmizer,
        'epochs': args.epochs,
        'layer_dim': args.layer_dim,
        'n_layers': args.n_layers,
        'params': params,
        'acc': acc,
        'time_elapsed': time_elapsed,
        'ptime_elapsed': ptime_elapsed
    }

    with open(f'{args.dest}/nn_{os.getpid()}.pkl', 'wb') as f:
        pickle.dump(return_dict, f)