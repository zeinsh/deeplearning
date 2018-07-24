#!/usr/bin/env python
# -*- coding: utf-8 -*-
# version 1
import os
import codecs

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
