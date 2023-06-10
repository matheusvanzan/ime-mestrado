from pathlib import Path
from django.shortcuts import render
import os
import evaluate
from sklearn.metrics import confusion_matrix

from collections import OrderedDict


PATHS = [
    'C:\\Users\\vanza\\Documents\\Codes\\ime\\am-malware\\data',
    'E:\\Backup Matheus\\IME'
]

def _metrics(predictions, references):
    good = 0
    bad = 0
    for pred, ref in zip(references, predictions):
        if pred == ref: good += 1
        else: bad += 1

    # print(' - GOOD:', good)
    # print(' - BAD:', bad)

    m = {
        'good': good,
        'bad': bad,
        'accuracy': round(100*good/(good+bad), 2)
    }
    # m.update(evaluate.load('accuracy').compute(
    #     predictions = predictions, 
    #     references = references
    # ))
    # m.update(evaluate.load('f1').compute(
    #     predictions = predictions, 
    #     references = references,
    #     average = None
    # ))
    return m

def metrics(path_results):
    predictions = []
    references = []
    with open(path_results, 'r') as f:
        for line in list(f.readlines())[1:]:
            items = line.split(',')
            pred = int(items[1].strip())
            ref = int(items[2].strip())
            predictions.append(pred)
            references.append(ref)                      

    # cm = confusion_matrix(references, predictions)
    # print(cm)
    m = _metrics(references, predictions) 
    return m


# Create your views here.
def index(request):

    model = 'gpt2'
    chunk = '32'
    batch = '160'

    limits = ['1024', '102400', 'all']
    epochs =  [1, 2, 3, 5, 10, 14, 15, 20] # range(21)
    folds = [str(x) for x in range(1, 10+1)] + ['k']
    k_fold = False

    version = '2'

    if 'model' in request.GET:
        model = request.GET['model']
    if 'limit' in request.GET:
        limits = [request.GET['limit']]
    if 'epochs' in request.GET:
        epochs = [request.GET['epochs']]
    if 'fold' in request.GET:
        if str(request.GET['fold']).isdigit(): # one fold
            folds = [request.GET['fold']]
        else: # k fold
            k_fold = True

    results = OrderedDict({})
    avg = 0
    for path in PATHS:
        for limit in limits:
            for epoch in epochs:

                m_kfold = {}
                for fold in folds:

                    path_ = os.path.join(path, model)
                    path_key = 'all.limit-{}.fold-{}.chunk-{}.epochs-{}.batch-{}.version-{}'.format(limit, fold, chunk, epoch, batch, version)
                    key = '{}.{}.{}.{}.{}.{}.{}'.format(model, limit, fold, chunk, epoch, batch, version)

                    if key not in results:
                        results.update({
                            key: {
                                'model': model,
                                'limit': limit,
                                'epoch': epoch,
                                'fold': fold,
                                'chunk': chunk,
                                'batch': batch,
                                'version': version,
                            }
                        })
                    
                    path_label = os.path.join(path_, path_key)
                    path_results = os.path.join(path_label, 'results.csv')

                    # print('path', path_results)

                    try:
                        m = metrics(path_results)
                        results[key].update(m)

                        for k in m:
                            if k not in m_kfold:
                                m_kfold.update({ k: [] })
                            m_kfold[k].append(m[k])

                        avg += m['accuracy']
                    except Exception as e:
                        # print(e)
                        pass

                # k-fold
                print('K-Fold', model, limit, epoch, m_kfold)
                m_kfold_avg = dict.fromkeys(m_kfold.keys(), 0)
                for k in m_kfold.keys():
                    m_kfold_avg[k] = round(sum(m_kfold[k])/len(m_kfold[k]), 2)
                results[key].update(m_kfold_avg)
                    

    for key, value in results.items():
        print(key, value)

    if k_fold: # pass only fold k
        items = [r for k, r in results.items() if 'k' in k]
    else:
        items = [r for _, r in results.items()]

    data = { 
        'results': items,
        'average': sum([i.get('accuracy', 0) for i in items])/sum([1 if i.get('accuracy', 0) != 0 else 0 for i in items]), 
        'count': len(items)
    }
    return render(request, 'main/index.html', data)