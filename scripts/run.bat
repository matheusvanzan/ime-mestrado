
set PATH=/c/Users/vanza/Documents/Codes/ime/am-malware/project

@set "PATH=C:\Users\vanza\Documents\Codes\ime\am-malware\project"


@REM for /l %%x in (1, 1, 10) do (
@REM     %PATH%\env\Scripts\python %PATH%\main.py --train --multi --model=gpt2 --limit=1024 --epochs=1 --batch=160 --fold=%%x --version=3
@REM     %PATH%\env\Scripts\python %PATH%\main.py --test --multi --model=gpt2 --limit=1024 --epochs=1 --batch=160 --fold=%%x --version=3
@REM )



%PATH%\env\Scripts\python %PATH%\main.py --train --multi --model=gpt2 --limit=all --epochs=6 --batch=160 --fold=2 --version=2
%PATH%\env\Scripts\python %PATH%\main.py --test  --multi --model=gpt2 --limit=all --epochs=6 --batch=160 --fold=2 --version=2