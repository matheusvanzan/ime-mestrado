import os
import shutil

PATHS = [
    'C:\\Users\\vanza\\Documents\\Codes\\ime\\am-malware\\data',
    'E:\\Backup Matheus\\IME'
]

def zip_partial(path):

    # TODO: check if results.csv exists

    if os.path.exists(path):
        print(f'Zip - {path}')
    else:
        return False

    path_dir = os.path.join(path, 'partial')
    path_zip = os.path.join(path, 'partial.zip') 
    
    if not os.path.exists(path_zip):
        shutil.make_archive(path_dir, 'zip', path_dir) # nao precisa do .zip, shutil faz zip do conteudo

    if os.path.exists(path_zip) and os.path.exists(path_dir):
        shutil.rmtree(path_dir)



def unzip_partial(path):
    pass


if __name__ == '__main__':

    models = ['gpt2']
    limits = ['1024', '102400', 'all']
    folds = [str(x) for x in range(1, 10+1)]
    chunks = ['32']
    epochs =  [str(x) for x in range(1, 20+1)]
    batchs = ['160']

    for PATH in PATHS:
        for model in models:
            for limit in limits:
                for fold in folds:
                    for chunk in chunks:
                        for epoch in epochs:
                            for batch in batchs:                                
                                path_name = 'all.limit-{}.fold-{}.chunk-{}.epochs-{}.batch-{}'.format(
                                    limit, fold, chunk, epoch, batch)
                                
                                path_dir = os.path.join(PATH, model, path_name)
                                zip_partial(path_dir)
