'''
Projeto de PAA

Analisar o gasto em tempo e memoria de um processo

Classificacao de caracteres utilizando redes neurais
'''

import os
import subprocess
import pickle
import pandas as pd
import statistics

from time import sleep

def doubles(first, last):
    value = first
    while value <= last:
        value *= 2
        yield value

def main():

    paa_v = 'paa-v3'
    load_mem_cost = 350 # MB
    
    trials = 3
    samples = 10
    optmizers = ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam']

    # v3
    defaults = { 'epochs': 100, 'layer_dims': 256, 'n_layers': 1 }
    epochs = range(1000, 5000+1, 1000) 
    layer_dims = range(1000, 5000+1, 1000)
    n_layers = range(100, 500+1, 100) # 100 eh o limite

    # v4
    # defaults = { 'epochs': 10, 'layer_dims': 10, 'n_layers': 1 }
    # epochs = range(10, 100+1, 10)
    # layer_dims = range(10, 100+1, 10)
    # n_layers = range(1, 10+1, 1)

    values = []
    for o in optmizers:
        values += [(o, e, defaults['layer_dims'], defaults['n_layers']) for e in epochs]
        values += [(o, defaults['epochs'], d, defaults['n_layers']) for d in layer_dims]
        values += [(o, defaults['epochs'], defaults['layer_dims'], n) for n in n_layers]

    print(values) 

    ######################################################################################
    #
    # Overwrite values for tests
    # 
    # (optmizer, epochs, layer_dim, n_layers)
    # 
    # values = [
    #     ('SGD', 10, 10, 10),
    #     ('SGD', 100, 1200, 1),
    #     ('SGD', 100, 2000, 1),
    #     ('SGD', 100, 256, 500),
    # ]

    features = ['optmizer', 'epochs', 'layer_dim', 'n_layers', 'params', \
        'acc', 'time_elapsed', 'ptime_elapsed', 'mem_max']

    total = len(values)
    for i, value in enumerate(values):
        optmizer, epochs, layer_dim, n_layers = value

        filename = f'{optmizer}_e_{epochs}_d_{layer_dim}_n_{n_layers}'
        filepath = f'/home/docky/data/{paa_v}/{filename}.csv'

        if os.path.exists(filepath):
            print(f'+ {i+1} of {total}: skip {filename}')
        else:
            print(f'+ {i+1} of {total}: start {filename}')

            costs = {'time_elapsed': [], 'ptime_elapsed': [], 'mem_max': []}
            for j in range(trials):
                # start NN process
                process_nn = subprocess.Popen([
                    '/home/docky/env/bin/python', '/home/docky/code/paa/mnist.py',
                    '--dest', f'/home/docky/data/tmp',
                    '--optmizer', str(optmizer),
                    '--epochs', str(epochs),
                    '--layer_dim', str(layer_dim),
                    '--n_layers', str(n_layers)
                ]) 

                # start memory process
                df_mem_path = f'/home/docky/data/tmp/mem_{process_nn.pid}.csv'
                process_mem = subprocess.Popen([
                    '/home/docky/env/bin/python', '/home/docky/code/mem.py',
                    '--dest', df_mem_path,
                    '--pid', str(process_nn.pid)
                ])

                process_nn.communicate() # blocks until complete

                # get NN data
                nn_path = f'/home/docky/data/tmp/nn_{process_nn.pid}.pkl'
                data = None
                while not isinstance(data, dict):
                    sleep(1)
                    with open(nn_path, 'rb') as f:
                        data = pickle.load(f)

                # get mem data
                df_mem = None
                while not isinstance(df_mem, pd.DataFrame):
                    sleep(1)
                    df_mem = pd.read_csv(df_mem_path, index_col=0)

                df_mem['mem'] = df_mem['mem'].div(1024**2).round(2) # MB
                data.update({'mem_max': df_mem['mem'].max() - load_mem_cost})

                costs['time_elapsed'].append(data['time_elapsed'])
                costs['ptime_elapsed'].append(data['ptime_elapsed'])
                costs['mem_max'].append(data['mem_max'])

                print(f'  - {j+1} of {trials}:' + \
                    ','.join([f' {f} = {data[f]}' for f in features]))

            # print(costs)
            data['time_elapsed'] = statistics.median(costs['time_elapsed'])
            data['ptime_elapsed'] = statistics.median(costs['ptime_elapsed'])
            data['mem_max'] = statistics.median(costs['mem_max'])

            print(f'+ {i+1} of {total}:' + \
                ','.join([f' {f} = {data[f]}' for f in features]))

            items = [str(data[f]) for f in features]
            with open(filepath, 'w') as f:
                f.write(','.join(items) + '\n')


if __name__ == '__main__':
    main()