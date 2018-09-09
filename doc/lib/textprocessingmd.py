#!/usr/bin/env python
# -*- coding: utf-8 -*-
# version 1
"""
text conversion functions added
- convert_puncs
- convert_escapes
- restore_escapes
- convert_text
"""

import numpy as np
import nltk
import string
nltk.download('stopwords')
from nltk.corpus import stopwords
import re

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')
STOPWORDS = set(stopwords.words('english'))

def text_prepare(text, lower=True, remove_stops=True):
    """ version 0
        text: a string
        
        return: modified initial string
    """
    if lower: text = text.lower()
    text = re.sub(REPLACE_BY_SPACE_RE,' ',text)
    text = re.sub(BAD_SYMBOLS_RE,'',text)
    if remove_stops: text = ' '.join(w for w in text.split() if w not in STOPWORDS)
    return text

#### text conversions #####
def convert_puncs(text):
    """ version 1
    put spaces before and after each punctuation, for tokenization
    """
    ret=text
    for char in string.punctuation:
        ret=ret.replace(char,' {}'.format(char))
    return ret
def convert_escapes(text):
    """ version 1
    replace tabs and endl with special tokens
    """
    ret=text
    ret=ret.replace('\t',' #tab ')
    ret=ret.replace('\n',' #endl ')
    return ret
def restore_escapes(text):
    """ version 1
    restore the state of space tokens
    """
    ret=text
    ret=ret.replace(' #tab ','\t')
    ret=ret.replace('#endl','\n>>>>')
    return ret
def convert_text(text,puncs=True,escapes=True):
    """ version 1
    combine conversions in one function
    """
    ret=text
    if puncs:ret=convert_puncs(ret)
    if escapes:ret=convert_escapes(ret)
    
    return ret
def restore_text(text,punc=True, escapes=True):
    ret=text
    #if puncs:ret=convert_puncs(ret)
    if escapes:ret=restore_escapes(ret)
    return ret

if __name__ == "__main__":

    text="This is a test example.\ttab\n"
    converted=convert_text(text)
    print("Original:{}\nConverted:{}\n".format(text,converted))
