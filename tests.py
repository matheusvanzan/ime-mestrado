import configparser
from settings import *


def test_imports():

    import pandas as pd
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)

    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    import torch
    torch.manual_seed(42)



def test_settings():
    print('PATH_GPT', PATH_GPT)
    print('PATH_PROJECT', PATH_PROJECT)
    print('PATH_DATA', PATH_DATA)



if __name__ == '__main__':
    test_settings()
    test_imports()