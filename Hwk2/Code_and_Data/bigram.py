#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# --------------------------------------------------
# Description:  A starter code
# --------------------------------------------------
# Author: Wang-SongSheng <wang.songsheng@connect.um.edu.mo>
# Created Date : March 14th 2021, 13:00:00
# --------------------------------------------------

import argparse
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from math import log, exp


def preprocess(inputfile, tokenized_file ='tokenized.txt'):
    try:
        with open(inputfile, 'r', encoding='utf-8') as inputfile:
            with open(tokenized_file, 'w', encoding='utf-8') as outfile:
                for line in inputfile:
                    line = line.lower().strip()
                    tokens = word_tokenize(line)
                    outfile.write('<s> ')
                    for word in tokens:
                        outfile.write(f"{word} ")
                    outfile.write("<\\s>\n")
    except FileNotFoundError:
        print(f"ERROR：input file {inputfile} not found!")
    return
def sentence_preprocess(sentence:str, word_dict):
    #TODO: preprocess the sentence string input from command line, or the test set sentence
    #   input: a string sentence, word dictionary
    #   output: the tokenized sentence (a list, each item corresponds to a word or punctuation of the sentence)
    #   Remember to lowercase all letters
    #   Remember to mask the word that didn't appear in the training set as <UNK>
    #   Remember to add the <s> and </s> tokens
    sentence = sentence.lower().strip()
    tokens = word_tokenize(sentence)
    sentence = [word if word in word_dict else "<UNK>" for word in tokens]
    sentence = ['<s>'] + sentence + ['</s>']
    return sentence
def count_word(inputfile, outputfile = 'word.txt'):
    #TODO: count the words from the corpus, and output the result to the output file in the format required.
    #   A list object may help you with this work.
    try:
        preprocess(inputfile)
        with open('tokenized.txt', 'r', encoding='utf-8') as tokenized_file:
            word_counter = defaultdict(int)
            for line in tokenized_file:
                for word in line.split():
                    word_counter[word] += 1
        with open(outputfile, 'w', encoding='utf-8') as outfile:
            with open('word.txt', 'w', encoding='utf-8') as wordfile:
                for word, count in word_counter.items():
                    outfile.write(f"{word} {count}\n")
                    if outputfile != 'word.txt':
                        wordfile.write(f"{word} {count}\n")
    except FileNotFoundError:
        print(f"ERROR：input file {tokenized_file} not found!")
    return
def count_bigram(inputfile, outputfile = 'bigram.txt'):
    # TODO: count the bigrams from the corpus, and output the result to the output file in the format required.
    #   You can use a string to represent a bigram
    #   A list object may help you with this work.
    try:
        preprocess(inputfile)
        with open('tokenized.txt', 'r', encoding='utf-8') as tokenized_file:
            bigram_counter = defaultdict(int)
            for line in tokenized_file:
                prev_word = ''
                for word in line.split():
                    if word != '<s>':
                        bigram = f"{prev_word} {word}"
                        bigram_counter[bigram] += 1
                    prev_word = word
        with open(outputfile, 'w', encoding='utf-8') as outfile:
            with open('bigram.txt', 'w', encoding='utf-8') as bigramfile:
                for bigram, count in bigram_counter.items():
                    outfile.write(f"{bigram} {count}\n")
                    if outputfile != 'word.txt':
                        bigramfile.write(f"{bigram} {count}\n")
    except FileNotFoundError:
        print(f"ERROR：input file {tokenized_file} not found!")
    return
def read_word_count(inputfile = 'word.txt'):
    #TODO: implement a tool function to read the stored word count
    #returns a dictionary, where word as the query and its frequency as the key. {'word0':1,'word1':2...}
    word_dict = defaultdict(int)
    with open(inputfile, 'r', encoding='utf-8') as corpus:
        for line in corpus:
            word, count = line.split()
            count = int(count)
            word_dict[word] = count
    return word_dict
def read_bigram_count(inputfile = 'bigram.txt'):
    #TODO: implement a tool function to read the stored bigram count
    #returns a dictionary, where bigram as the query and its frequency as the key. {'word0':1,'word1':2...}
    bigram_dict = defaultdict(int)
    with open(inputfile, 'r', encoding='utf-8') as corpus:
        for line in corpus:
            word0, word1, count = line.split()
            count = int(count)
            bigram = f"{word0} {word1}"
            bigram_dict[bigram] = count
    return bigram_dict
def add_one_perplexity(sentence, word_dict = None, bigram_dict = None):
    #TODO: calculate the perplexity based on the add-1 smoothing
    return add_n_perplexity(sentence, 1, word_dict, bigram_dict)
def add_n_perplexity(sentence, n, word_dict = None, bigram_dict = None):
    #TODO: calculate the perplexity based on the add-n smoothing
    if word_dict is None:
        word_dict = read_word_count()
    if bigram_dict is None:
        bigram_dict = read_bigram_count()
    sentence = sentence_preprocess(sentence, word_dict)
    prev_word = ''
    v = len(word_dict)
    log_prob = 0.0 # log(1.0)
    for word in sentence:
        if word != '<s>':
            numerator = bigram_dict[f"{prev_word} {word}"] + n
            denominator = word_dict[prev_word] + n * v
            log_prob += log(numerator / denominator)
        prev_word = word
    perplexity = exp(-log_prob / len(sentence))
    return perplexity
def add_n_perplexity_batch(input, output, n):
    #TODO:
    #   Read the test-set from the input file, do the preprocessing using the sentence_preprocess function for each sentence
    #   Calculate the perplexity of each sentence based on the add-n smoothing in batch mode
    #   Calculate the average perplexity of the whole test-set
    #   Output the experiment result in the format required
    word_dict = read_word_count()
    bigram_dict = read_bigram_count()
    cnt_sentence = 0.0
    sum_ppl = 0.0
    with open(input, 'r', encoding='utf-8') as test_set:
        with open(output, 'w', encoding='utf-8') as output_file:
            output_file.write("remained for avg ppl\n")
            for line in test_set:
                if line.endswith('\n'):
                    line = line[:-1]
                ppl = add_n_perplexity(line, n, word_dict, bigram_dict)
                output_file.write(f"{line} {f"{ppl:.2f}" if ppl != float('inf') else 'INF'}\n")
                if ppl != float('inf'):
                    cnt_sentence += 1.0
                    sum_ppl += ppl

    avg_ppl = sum_ppl / cnt_sentence
    with open(output, 'r', encoding='utf-8') as output_file:
        lines = output_file.readlines()
    if lines:  # 确保文件不为空
        lines[0] = f"Test-Set-PPL: {avg_ppl:.2f}\n"
    with open(output, 'w', encoding='utf-8') as output_file:
        output_file.writelines(lines)
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
    # main()
    # count_word('my_test_train.txt')
    # count_bigram('my_test_train.txt')
    # add_n_perplexity_batch('my_test_test.txt', 'my_final_result.txt', 2)

    # count_word('news.train')
    # count_bigram('news.train')
    n = 3
    add_n_perplexity_batch('news.test', f'perplexity-{n}.txt', n)