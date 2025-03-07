#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# --------------------------------------------------
# Description:  A starter code
# --------------------------------------------------
# Author: Wang-SongSheng <wang.songsheng@connect.um.edu.mo>
# Created Date : March 14th 2021, 13:00:00
# --------------------------------------------------

import argparse

def preprocess(inputfile,outputfile):
    #TODO: preprocess the input file, and output the result to the output file.
    #   Convert all letters to the lowercase
    #   Add the <s> and </s> tokens
    #   Use NLTK.word_tokenize() to tokenize the sentence
    #   The format of the output file should be txt, each word or Punctuation is separated by a blank.
    return
def sentence_preprocess(sentence,word_dict):
    #TODO: preprocess the sentence string input from command line, or the test set sentence
    #   input: a string sentence, word dictionary
    #   output: the tokenized sentence (a list, each item corresponds to a word or punctuation of the sentence)
    #   Remember to lowercase all letters
    #   Remember to mask the word that didn't appear in the training set as <UNK>
    #   Remember to add the <s> and </s> tokens
    return sentence
def count_word(inputfile,outputfile):
    #TODO: count the words from the corpus, and output the result to the output file in the format required.
    #   A list object may help you with this work.
    return
def count_bigram(inputfile,outputfile):
    # TODO: count the bigrams from the corpus, and output the result to the output file in the format required.
    #   You can use a string to represent a bigram
    #   A list object may help you with this work.
    return
def read_word_count():
    word_dict = {}
    #TODO: implement a tool function to read the stored word count
    #returns a dictionary, where word as the query and its frequency as the key. {'word0':1,'word1':2...}
    return word_dict
def read_bigram_count():
    bigram_dict = {}
    #TODO: implement a tool function to read the stored bigram count
    #returns a dictionary, where bigram as the query and its frequency as the key. {'word0':1,'word1':2...}
    return bigram_dict
def add_one_perplexity(sentence):
    word_dict = read_word_count()
    bigram_dict = read_bigram_count()
    sentence = sentence_preprocess(sentence,word_dict)
    #TODO: calculate the perplexity based on the add-1 smoothing
    return perplexity
def add_n_perplexity(sentence,n):
    word_dict = read_word_count()
    bigram_dict = read_bigram_count()
    sentence = sentence_preprocess(sentence,word_dict)
    #TODO: calculate the perplexity based on the add-n smoothing
    return perplexity
def add_n_perplexity_batch(input,output,n):
    word_dict = read_word_count()
    bigram_dict = read_bigram_count()
    #TODO:
    #   Read the test-set from the input file, do the preprocessing using the sentence_preprocess function for each sentence
    #   Calculate the perplexity of each sentence based on the add-n smoothing in batch mode
    #   Calculate the average perplexity of the whole test-set
    #   Output the experiment result in the format required
    return
def main():
    ''' Main Function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-pps', '--preprocess',type=str,nargs=2,help='preprocess the dataset')
    parser.add_argument('-cw','--count_word',type=str,nargs=2,help='count the words from the corpus')
    parser.add_argument('-cb','--count_bigram',type=str,nargs=2,help='count the bigrams from the corpus')
    parser.add_argument('-ppl1','--add_one_perplexity',type=str,nargs=1,
                        help='calculate the perplexity of the sentence using the add-1 smoothing')
    parser.add_argument('-ppln','--add_n_perplexity',type=str,nargs=2,
                        help='calculate the perplexity of the sentence using the add-n smoothing')
    parser.add_argument('-pplnb','--add_n_perplexity_batch', type=str, nargs=3,
                        help='calculate the perplexity of the sentence using the add-n smoothing')
    opt=parser.parse_args()

    if(opt.preprocess):
        input_file = opt.preprocess[0]
        output_file = opt.preprocess[1]
        preprocess(input_file,output_file)
    elif(opt.count_word):
        input_file = opt.count_word[0]
        output_file = opt.count_word[1]
        count_word(input_file,output_file)
    elif(opt.count_bigram):
        input_file = opt.count_bigram[0]
        output_file = opt.count_bigram[1]
        count_bigram(input_file,output_file)
    elif(opt.add_one_perplexity):
        ppl = add_one_perplexity(opt.add_one_perplexity[0])
        print('The perplexity of the sentence using the add-one smooting is: '+str(ppl))
    elif(opt.add_n_perplexity):
        sentence = opt.add_n_perplexity[0]
        n = opt.add_n_perplexity[1]
        ppl = add_n_perplexity(sentence,n)
        print('The perplexity of the sentence using the add-'+str(n)+' smooting is: ' + str(ppl))
    elif(opt.add_n_perplexity_batch):
        input = opt.add_n_perplexity_batch[0]
        output = opt.add_n_perplexity_batch[1]
        n = int(opt.add_n_perplexity_batch[2])
        add_n_perplexity_batch(input, output, n)



if __name__ == '__main__':
    import os
    main()
