# version 0
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')
STOPWORDS = set(stopwords.words('english'))

def text_prepare(text, lower=True, remove_stops=True):
    """
        text: a string
        
        return: modified initial string
    """
    if lower: text = text.lower()
    text = re.sub(REPLACE_BY_SPACE_RE,' ',text)
    text = re.sub(BAD_SYMBOLS_RE,'',text)
    if remove_stops: text = ' '.join(w for w in text.split() if w not in STOPWORDS)
    return text
