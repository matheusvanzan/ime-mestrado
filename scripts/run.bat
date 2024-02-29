
set PATH=/c/Users/vanza/Documents/Codes/ime/am-malware/project

@set "PATH=C:\Users\vanza\Documents\Codes\ime\am-malware\project"


for /l %%x in (1, 1, 10) do (
    %PATH%\env\Scripts\python %PATH%\main.py --train --multi --model=gpt2 --limit=all --chunk=32 --batch=160 --fold=%%x --epochs=4 --version=3
    %PATH%\env\Scripts\python %PATH%\main.py --test  --multi --model=gpt2 --limit=all --chunk=32 --batch=160 --fold=%%x --epochs=4 --version=3
)
