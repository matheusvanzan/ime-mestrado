#!/bin/bash

PATH=/home/utilizador/codes/project

for i in {1..10}; do
    echo "$i"
    
    $PATH/env/bin/python $PATH/main.py --train --multi --model=gpt2 --limit=2048 --epochs=1 --batch=16 --fold=$i --version=3

    $PATH/env/bin/python $PATH/main.py --test  --multi --model=gpt2 --limit=2048 --epochs=1 --batch=16 --fold=$i --version=3
done