#!/usr/bin/env python
# -*- coding: utf-8 -*-
# version 1.5
## add log
import os
import codecs
from utilitiesmd import getDatetimeStr

def getFilesFromPath(path):
    return os.listdir(path)
def readTxtFromFile(path,filename):
    try:
        with codecs.open(path+'/'+filename,'r',encoding='utf8') as f:
            text = f.read()
            return text
    except:
        print("{} not opened".format(filename))
        return ""

# log functions
def log(path,filename,message):
    dt=getDatetimeStr()
    with codecs.open(path+'/'+filename,'a',encoding='utf8') as f:
        f.write('\n----{}----\n{}\n'.format(dt,message))
def clearLog(path,filename):
    codecs.open(path+'/'+filename,'w',encoding='utf8')

        