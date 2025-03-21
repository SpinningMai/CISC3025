#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# --------------------------------------------------
# Description:  Template Code For PreProcessing
# --------------------------------------------------
# Author: Wang-SongSheng <wang.songsheng@connect.um.edu.mo>
# Created Date : Feb 21st 2021, 17:08:00
# --------------------------------------------------

import re
import json
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
from tqdm import tqdm


def read_files(dir, fileList):
    #read the files based on the file list, output a list of string as the corpus
    corpus = []
    for item in fileList:
        with open(dir + '/' + item, encoding='utf-8') as f:
            corpus.append([item.strip('.txt'),f.read()])
    return corpus
def preprocess_corpus(corpus):
    for i in tqdm(range(len(corpus))):
        #strip some symbol from the original text
        stripped_text = text_strip(corpus[i][1])
        #use the tokenize function to tokenize the text
        tokenized_text = text_tokenize(stripped_text)
        #use the stemmer to stem the tokenized words
        for j in range(len(tokenized_text)):
            tokenized_text[j] = text_stem(tokenized_text[j])
        corpus[i][1] = tokenized_text
    return corpus
def text_strip(text):
    #DONE:Complete the function to finish the following task:
    text = re.sub(pattern=r'\.\.\.more$', repl='', string=text)
    text = re.sub(pattern=r'[\s]+',repl=' ',string=text)
    text = text.lower()
    return text
def text_tokenize(text) -> list:
    #DONE: Use the NLTK package or regular expression and string operations to tokenize the text
    tokens = word_tokenize(text)
    #The function should return a list that contains the tokenized words as the elements.
    return tokens
def text_stem(word:str) -> str:
    #DONE: Use the stemmer to stem the tokenized word
    #The function receives a word that needs to be stemmed and returns the stemmed word.
    stemmer = PorterStemmer()
    stemmed_word = stemmer.stem(word)
    return stemmed_word
def output_to_file(corpus,save_dir):
    for item in corpus:
        with open(save_dir+item[0]+'.txt','w', encoding='utf-8') as f:
            f.write(str(item[1]))
            f.close()
def output_to_json(corpus,save_dir):
    with open(save_dir+'corpus.json','w', encoding='utf-8') as f:
        #DONE: Use JSON package to dump the corpus to corpus.json file
        f.write(json.dumps(corpus))
        f.close()
def main():
    ''' Main Function '''
    import os
    fileList = os.listdir('test')
    save_dir = './corpus/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    corpus = read_files('test',fileList)
    preprocessed_corpus = preprocess_corpus(corpus)
    output_to_file(preprocessed_corpus,save_dir)
    output_to_json(preprocessed_corpus,'./')

if __name__ == '__main__':
    main()
