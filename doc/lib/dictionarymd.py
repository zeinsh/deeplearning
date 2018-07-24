#!/usr/bin/env python
# -*- coding: utf-8 -*-
# version 1

from collections import Counter, defaultdict
import os
import codecs

class Dictionary:
    def __init__(self):
        self.idx2word=None
        self.word2idx=None
    def make_vocab(self,text,fpath,fname):
        '''Constructs vocabulary.

        Args:
          fpath: A string. Input file path.
          fname: A string. Output file name.

        Writes vocabulary line by line to `preprocessed/fname`
        '''  
        words = text.split()
        word2cnt = Counter(words)
        if not os.path.exists('preprocessed'): os.mkdir('preprocessed')
        with codecs.open('{}/{}'.format(fpath,fname), 'w', 'utf-8') as fout:
            fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000".format(
                    "<PAD>", "<UNK>" , "<START>",'.'))
            for word, cnt in word2cnt.most_common(len(word2cnt)):
                fout.write(u"{}\t{}\n".format(word, cnt))
    def load_vocab(self,path,filename):
        vocab = [line.split()[0] for line in codecs.open(path+'/'+filename, 'r', 'utf-8').read().splitlines()]
        vocab = vocab[:10000]
        self.word2idx = defaultdict(lambda:1,{word: idx for idx, word in enumerate(vocab)})
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}
    def text2vec(self,text):
        return [self.word2idx["<START>"]]+[self.word2idx[token] for token in text.split()]
    def vec2text(self,vec):
        return ' '.join([self.idx2word[idx] for idx in vec])

if __name__ == "__main__":
    fpath='../temp'
    fname='test.voc'
    text="this is test text\n"
    
    dictionary=Dictionary()
    dictionary.make_vocab(text,fpath,fname)
    dictionary.load_vocab(fpath,fname)
    
    vec=dictionary.text2vec(text)
    ret=dictionary.vec2text(vec)
    
    print(
"""Original text:\"{}\"\n
Restored text:\"{}\"\n
Vector:{}
""".format(text,ret,vec))
