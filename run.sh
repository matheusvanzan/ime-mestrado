#!/bin/bash

PATH=/home/utilizador/codes/project


$PATH/env/bin/python $PATH/main.py --train --multi --model=gpt2 --limit=512 --epochs=1 --batch=16 --fold=1 --version=3
$PATH/env/bin/python $PATH/main.py --test --multi --model=gpt2 --limit=512 --epochs=1 --batch=16 --fold=1 --version=3