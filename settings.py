#!/usr/bin/env python
# coding: utf-8

import os

# NAMES
PROJECT_NAME = 'project'

# PATHS
PATH_PROJECT = '/content/drive/Shareddrives/GPTJ/project/'
PATH_DATA = '/content/drive/Shareddrives/GPTJ/data/'

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


OPCODES = 'aaa,aad,aam,aas,adc,add,and,arpl,bound,bsf,bsr,bswap,bt,btc,btr,' +           'bts,call,cbw,cdq,clc,cld,cli,clts,cmc,cmp,cmps,cmpxchg,cwd,cwde,' +           'daa,das,dec,div,enter,esc,fwait,hlt,idiv,imul,in,inc,ins,int,' +           'into,invd,invlpg,iret,iretd,ja,jae,jb,jbe,jc,jcxz,je,jecxz,jg,j' +           'ge,jl,jle,jmp,jna,jnae,jnb,jnbe,jnc,jne,jng,jnge,jnl,jnle,jno,' +           'jnp,jns,jnz,jo,jp,jpe,jpo,js,jz,lahf,lar,lds,lea,leave,les,lfs,' +           'lgdt,lgs,lidt,lldt,lmsw,lock,lods,loop,loope,loopne,loopnz,' +           'loopz,lsl,lss,ltr,mov,movs,movsx,movzx,mul,neg,nop,not,or,out,' +           'outs,pop,popa,popad,popf,popfd,push,pusha,pushad,pushf,pushfd,' +           'rcl,rcr,rep,repe,repne,repnz,repz,ret,retf,rol,ror,sahf,sal,sar,' +           'sbb,scas,setae,setb,setbe,setc,sete,setg,setge,setl,setle,setna,' +           'setnae,setnb,setnc,setne,setng,setnge,setnl,setnle,setno,setnp,' +           'setns,setnz,seto,setp,setpe,setpo,sets,setz,sgdt,shl,shld,shr,' +           'shrd,sidt,sldt,smsw,stc,std,sti,stos,str,sub,test,verr,verw,' +           'wait,wbinvd,xchg,xlat,xlatb,xor,ah,al,ax,bh,bl,bp,bx,ch,cl,cs,' +           'cx,dh,di,dl,ds,dx,eax,ebp,ebx,ecx,edi,edx,eflags,eip,es,esi,esp' +           ',fs,gs,ip,si,sp,ss'
NPL_VOCAB = set(OPCODES.split(','))