from evaluate import load
from sklearn.metrics import confusion_matrix


def get_metrics(references, predictions):

    equals = [int(r == f) for r, f in zip(references, predictions)]
    good = sum(equals)
    bad = len(references) - good

    cm = confusion_matrix(references, predictions)

    metrics = {
        'good': good,
        'bad': bad,
        'matrix': cm
    }

    metrics.update(load('accuracy').compute(
        predictions = predictions, 
        references = references
    ))
    metrics.update(load('precision').compute(
        predictions = predictions, 
        references = references,
        average = None
    ))
    # metrics.update(load('recall').compute(
    #     predictions = predictions, 
    #     references = references,
    #     average = None
    # ))
    metrics.update(load('f1').compute(
        predictions = predictions, 
        references = references,
        average = None
    ))

    return metrics