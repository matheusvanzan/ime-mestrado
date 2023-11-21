# gpt-malware

## Install

```
$ virtualenv env
$ env/Scripts/activate
>>> pip install -r requirements.txt
```

Cria o virtualenv e instala todas as dependências.

## Processing

```
>>> python main.py --process 

```

- Pré-processamento dos arquivos .asm de `data/kaggle/asm` para `data/kaggle/proc-1/all`.
- Divisão em labels para `data/kaggle/proc-1/1`, `data/kaggle/proc-1/2`, ...

## Dataset Cache

```
>>> python main.py --cache --fold=1 --version=3

```

- Tokenização dos arquivos de .asm para .csv de tokens
- Criação do banco com os aparametros default


# Linux
sudo apt-get install libmkl-avx
sudo apt-get install libmkl-avx2