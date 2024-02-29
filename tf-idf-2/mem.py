import os
import psutil
import pandas as pd

from time import sleep
from argparse import ArgumentParser


def get_mem(pid=None):
    if not pid:
        pid = os.getpid()

    process = psutil.Process(pid)

    mem = process.memory_info().rss # bytes
    mem_kb = round(mem/1024, 2)
    mem_mb = round(mem/1024**2, 2)
    mem_gb = round(mem/1024**3, 2)
    mem_str = ''

    if mem_kb < 1024: 
        mem_str = f'{mem_kb} KB'

    if 1024 <= mem_kb < 1024**2: 
        mem_str = f'{mem_mb} MB'
    
    if 1024**2 <= mem_kb < 1024**3: 
        mem_str = f'{mem_gb} GB'
    
    return (mem, mem_str)

def check_mem(pid=None):
    if not pid:
        pid = os.getpid()

    mem, mem_str = get_mem(pid)
    print(mem_str)

def is_running(pid):        
    '''
        Check For the existence of a unix pid.
        Sending signal 0 to a pid will raise an OSError exception if the pid is not running, and do nothing otherwise.
    '''

    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True

def main():
    pass

if __name__ == '__main__':
    # main()
    parser = ArgumentParser()
    parser.add_argument('-p', '--pid', dest='pid', help='Create dataset')
    parser.add_argument('-d', '--dest', dest='dest', help='')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='')
    args = parser.parse_args()
    # print(args)

    mem_list = []
    pid = int(args.pid)
    while is_running(pid):
        mem, mem_str = get_mem(pid = pid)
        mem_list.append(mem)

        if args.verbose:
            print(mem_str, len(mem_list))
        sleep(1) # secs

    df = pd.DataFrame(data=mem_list, columns=['mem'])

    if args.dest:
        df.to_csv(args.dest)
    else:
        df.to_csv(f'data/tmp/mem_{pid}.csv')

    if args.verbose:
        print('process killed')

