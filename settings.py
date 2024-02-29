#!/usr/bin/env python
# coding: utf-8

import os
import platform
from torch import NoneType

# NAMES
PROJECT_NAME = 'project'

# PATHS
print(platform.system())

if platform.system() == 'Windows':
    # PATH_GPT = 'C:\\Users\\vanza\\Documents\\Codes\\ime\\am-malware\\'
    PATH_GPT = 'D:\\IME\\gpt-malware\\'
    VERSION = 1
elif platform.system() == 'Linux':
    PATH_GPT = '/home/utilizador/codes/'
    VERSION = 2
else:
    raise Exception('Paths not configured in settings.py')

PATH_PROJECT = os.path.join(PATH_GPT, 'project')

# PATH_DATA = os.path.join(PATH_GPT, 'data-big2015')
# DATASET_CLASSES = [str(i) for i in range(9)]

PATH_DATA = os.path.join(PATH_GPT, 'data-malv2022')
DATASET_CLASSES = ['0', '1']

NN_LEN_CLASSES = len(DATASET_CLASSES)
NN_KFOLD = 10
NN_EPOCHS = 20
NN_BATCH = 32
NN_MEAN_TRIALS = 1

PATH_DATA_KAGGLE = os.path.join(PATH_DATA, 'dataset')
PATH_DATA_LABELS = os.path.join(PATH_DATA_KAGGLE, 'labels.csv')
PATH_DATA_LABELS_0 = os.path.join(PATH_DATA_KAGGLE, 'labels-0.csv')
PATH_DATA_ASM = os.path.join(PATH_DATA_KAGGLE, 'asm')
PATH_DATA_PROC_1 = os.path.join(PATH_DATA_KAGGLE, 'proc-1')
PATH_DATA_TFIDF = os.path.join(PATH_DATA, 'tf-idf')

# NPL
NPL_CHARS_TO_REMOVE = ''.join([
    "'",
    '"',
    '.,;:!?', # pontos
    '+-=*<>', # sinais
    '()[]{}', # fechamento
    '\n\r\t', # espacos
    '´^~`', # acentos
    '@#$%¨&|_/'
    '' # \\x19
])
NPL_WORDS_TO_REMOVE = [
    'text',
    's u b r o u t i n e',
    'code xref'
]

OPCODES = 'aaa,aad,aam,aas,adc,add,and,arpl,bound,bsf,bsr,bswap,bt,btc,btr,' + \
          'bts,call,cbw,cdq,clc,cld,cli,clts,cmc,cmp,cmps,cmpxchg,cwd,cwde,' + \
          'daa,das,dec,div,enter,esc,fwait,hlt,idiv,imul,in,inc,ins,int,' + \
          'into,invd,invlpg,iret,iretd,ja,jae,jb,jbe,jc,jcxz,je,jecxz,jg,j' + \
          'ge,jl,jle,jmp,jna,jnae,jnb,jnbe,jnc,jne,jng,jnge,jnl,jnle,jno,' + \
          'jnp,jns,jnz,jo,jp,jpe,jpo,js,jz,lahf,lar,lds,lea,leave,les,lfs,' + \
          'lgdt,lgs,lidt,lldt,lmsw,lock,lods,loop,loope,loopne,loopnz,' + \
          'loopz,lsl,lss,ltr,mov,movs,movsx,movzx,mul,neg,nop,not,or,out,' + \
          'outs,pop,popa,popad,popf,popfd,push,pusha,pushad,pushf,pushfd,' + \
          'rcl,rcr,rep,repe,repne,repnz,repz,ret,retf,rol,ror,sahf,sal,sar,' + \
          'sbb,scas,setae,setb,setbe,setc,sete,setg,setge,setl,setle,setna,' + \
          'setnae,setnb,setnc,setne,setng,setnge,setnl,setnle,setno,setnp,' + \
          'setns,setnz,seto,setp,setpe,setpo,sets,setz,sgdt,shl,shld,shr,' + \
          'shrd,sidt,sldt,smsw,stc,std,sti,stos,str,sub,test,verr,verw,' + \
          'wait,wbinvd,xchg,xlat,xlatb,xor,ah,al,ax,bh,bl,bp,bx,ch,cl,cs,' + \
          'cx,dh,di,dl,ds,dx,eax,ebp,ebx,ecx,edi,edx,eflags,eip,es,esi,esp,' + \
          'fs,gs,ip,si,sp,ss'
NPL_VOCAB = set(OPCODES.split(','))


# DATASET
DATASET_CHUNK_SIZE = 32 # size per line in .csv
DATASET_LIMIT = 'all' # number of tokens

# GPT
MODEL_NAME = 'gpt2'
GPT_TRAIN_CHUNK_SIZE = DATASET_CHUNK_SIZE
GPT_TEST_CHUNK_SIZE = DATASET_CHUNK_SIZE
GPT_BATCH_SIZE = 160
GPT_EPOCHS = 1

