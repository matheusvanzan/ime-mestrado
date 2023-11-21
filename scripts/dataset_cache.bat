
set PATH=/c/Users/vanza/Documents/Codes/ime/am-malware/project
@set "PATH=C:\Users\vanza\Documents\Codes\ime\am-malware\project"


for /l %%x in (1, 1, 10) do (
    %PATH%\env\Scripts\python %PATH%\main.py --cache --fold=%%x --version=3
)

for /l %%x in (1, 1, 10) do (
    %PATH%\env\Scripts\python %PATH%\main.py --cache --fold=%%x --version=2
)

@REM %PATH%\env\Scripts\python %PATH%\main.py --train --multi --model=gpt2 --limit=all --epochs=5 --batch=160 --fold=2 --version=3