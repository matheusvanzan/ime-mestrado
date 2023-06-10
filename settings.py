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
    PATH_GPT = 'C:\\Users\\vanza\\Documents\\Codes\\ime\\am-malware\\'
elif platform.system() == 'Linux':
    PATH_GPT = '/home/utilizador/codes/'
else:
    raise Exception('Paths not configured in settings.py')

PATH_PROJECT = os.path.join(PATH_GPT, 'project')
PATH_DATA = os.path.join(PATH_GPT, 'data')

PATH_DATA_KAGGLE = os.path.join(PATH_DATA, 'kaggle')
PATH_DATA_LABELS = os.path.join(PATH_DATA_KAGGLE, 'trainLabels.csv')
PATH_DATA_ASM = os.path.join(PATH_DATA_KAGGLE, 'asm')

PATH_DATA_PROC_1 = os.path.join(PATH_DATA_KAGGLE, 'proc-1')
PATH_DATA_PROC_2 = os.path.join(PATH_DATA_KAGGLE, 'proc-2')
PATH_DATA_PROC_3 = os.path.join(PATH_DATA_KAGGLE, 'proc-3')
PATH_DATA_PROC_4 = os.path.join(PATH_DATA_KAGGLE, 'proc-4')
PATH_DATA_PROC_5 = os.path.join(PATH_DATA_KAGGLE, 'proc-5')

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
DATASET_CLASSES = [str(i) for i in range(9)]
DATASET_CHUNK_SIZE = 32 # size per line in .csv
DATASET_LIMIT = 1024 # number of tokens

# GPT
MODEL_NAME = 'gpt2'
GPT_TRAIN_CHUNK_SIZE = DATASET_CHUNK_SIZE
GPT_TEST_CHUNK_SIZE = DATASET_CHUNK_SIZE
GPT_BATCH_SIZE = 160
GPT_EPOCHS = 2
VERSION = 1



