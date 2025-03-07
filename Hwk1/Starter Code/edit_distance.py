#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# --------------------------------------------------
# Description:  A starter code
# --------------------------------------------------
# Author: Wang-SongSheng <wang.songsheng@connect.um.edu.mo>
# Created Date : March 4th 2021, 12:00:00
# Student: Mai-JiaJun <dc12785@um.edu.mo>
# Last Edited Date : February 21st 2025, 22:00:00
# --------------------------------------------------

import argparse
from tqdm import tqdm
import re

def word_edit_distance(x:str, y:str, cost_ins:int = 1, cost_del:int = 1, cost_sus:int = 2) -> (int, list):
    table = [[['', float('inf')] for _ in range(len(y) + 1)] for _ in range(len(x) + 1)]
    table[0][0] = ['', 0]
    for i in range(1, len(x) + 1):
        table[i][0] = ['^', table[i - 1][0][1] + cost_del]
    for j in range(1, len(y) + 1):
        table[0][j] = ['<', table[0][j - 1][1] + cost_ins]

    for i in range(len(x)):
        for j in range(len(y)):
            sus_cost = table[i][j][1] + (0 if x[i] == y[j] else cost_sus)
            up_cost = table[i][j + 1][1] + cost_del
            left_cost = table[i + 1][j][1] + cost_ins

            if table[i + 1][j + 1][1] > sus_cost:
                table[i + 1][j + 1] = ['`', sus_cost]
            if table[i + 1][j + 1][1] > up_cost:
                table[i + 1][j + 1] = ['^', up_cost]
            if table[i + 1][j + 1][1] > left_cost:
                table[i + 1][j + 1] = ['<', left_cost]

    edit_distance = table[len(x)][len(y)][1]
    alignment = [[],[]]

    i, j = len(x), len(y)
    while not (i == 0 and j == 0):
        move = table[i][j][0]
        if move == '`':
            alignment[0].append(x[i - 1])
            alignment[1].append(y[j - 1])
            i -= 1
            j -= 1
        elif move == '<':
            alignment[0].append('-')
            alignment[1].append(y[j - 1])
            j -= 1
        elif move == '^':
            alignment[0].append(x[i - 1])
            alignment[1].append('-')
            i -= 1

    alignment[0].reverse()
    alignment[1].reverse()
    return edit_distance, alignment

def sentence_edit_distance(x:list, y:list, cost_ins:int = 1, cost_del:int = 1, cost_sus:int = 2) -> (int, list):
    table = [[['', float('inf')] for _ in range(len(y) + 1)] for _ in range(len(x) + 1)]
    table[0][0] = ['', 0]
    for i in range(1, len(x) + 1):
        table[i][0] = ['^', table[i - 1][0][1] + cost_del]
    for j in range(1, len(y) + 1):
        table[0][j] = ['<', table[0][j - 1][1] + cost_ins]

    for i in range(len(x)):
        for j in range(len(y)):
            sus_cost = table[i][j][1] + (0 if x[i] == y[j] else cost_sus)
            up_cost = table[i][j + 1][1] + cost_del
            left_cost = table[i + 1][j][1] + cost_ins

            if table[i + 1][j + 1][1] > sus_cost:
                table[i + 1][j + 1] = ['`', sus_cost]
            if table[i + 1][j + 1][1] > up_cost:
                table[i + 1][j + 1] = ['^', up_cost]
            if table[i + 1][j + 1][1] > left_cost:
                table[i + 1][j + 1] = ['<', left_cost]


    edit_distance = table[len(x)][len(y)][1]
    alignment = [[], []]

    i, j = len(x), len(y)
    while not (i == 0 and j == 0):
        move = table[i][j][0]
        if move == '`':
            alignment[0].append(x[i - 1])
            alignment[1].append(y[j - 1])
            i -= 1
            j -= 1
        elif move == '<':
            alignment[0].append('-')
            alignment[1].append(y[j - 1])
            j -= 1
        elif move == '^':
            alignment[0].append(x[i - 1])
            alignment[1].append('-')
            i -= 1

    alignment[0].reverse()
    alignment[1].reverse()
    return edit_distance, alignment

def sentence_preprocess(sentence:str) -> list[str]:
    words:list = sentence.split()
    no_sign_words:list = [
        re.sub(r"(?!^)[^\w\s\$\%\&\*\=\+\-\/\_'](?!$)", '', word)
        for word in words
    ]
    sentence:list = []
    for word in no_sign_words:
        if word != '':
            sentence.append(word)
    return sentence

def output_alignment(alignment):
    #output the alignment in the format required
    if len(alignment[0]) != len(alignment[1]):
        print('ERROR: WRONG ALIGNMENT FORMAT')
        input()
        exit(0)
    print('An possible alignment is:')
    merged_matrix = alignment[0] + alignment[1]
    max_len = 0
    for item in merged_matrix:
        if len(item) > max_len:
            max_len = len(item)
    for i in range(len(alignment[0])):
        print (alignment[0][i].rjust(max_len)+' ',end=''),
    print('')
    for i in range(len(alignment[0])):
        print (('|').rjust(max_len) + ' ',end=''),
    print('')
    for i in range(len(alignment[1])):
        print (alignment[1][i].rjust(max_len)+' ',end='')
    print('')
    return

def batch_word(inputfile,outputfile):
    try:
        with open(inputfile, 'r', encoding='utf-8') as inputfile:
            with open(outputfile, 'w', encoding='utf-8') as outfile:
                r_word, h_word = '', ''
                for line in tqdm(inputfile):
                    words = line.split()
                    if words[0] == 'R':
                        r_word = words[1]
                    else:
                        h_word = re.sub(r'\*[\d]+$', '', words[1]) # remove times suffix "*n"
                        edit_distance, _ = word_edit_distance(h_word, r_word)
                        line = f"{line[:-1]} {edit_distance}\n"
                    outfile.write(line)
    except FileNotFoundError:
        print(f"ERROR：input file {inputfile} not found!")

    return

def batch_sentence(inputfile,outputfile):
    try:
        with open(inputfile, 'r', encoding='utf-8') as inputfile:
            with open(outputfile, 'w', encoding='utf-8') as outfile:
                r_sentence, h_sentence = '', ''
                r_list, h_list = [], []
                for line in tqdm(inputfile):
                    if line[0] == 'R':
                        r_sentence = line[2:]
                        r_list = sentence_preprocess(r_sentence)
                    else:
                        h_sentence = line[2:]
                        h_list = sentence_preprocess(h_sentence)
                        edit_distance, _ = sentence_edit_distance(h_list, r_list)
                        line = f"{line[:-1]} {edit_distance}\n"
                    outfile.write(line)
    except FileNotFoundError:
        print(f"ERROR：input file {inputfile} not found!")

    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--word',type=str,nargs=2,help='word comparson')
    parser.add_argument('-s','--sentence',type=str,nargs=2,help='sentence comparison')
    parser.add_argument('-bw','--batch_word',type=str,nargs=2,help='batch word comparison,input the filename')
    parser.add_argument('-bs','--batch_sentence',type=str,nargs=2,help='batch word comparison,input the filename')

    opt=parser.parse_args()

    if(opt.word):
        edit_distance,alignment = word_edit_distance(opt.word[0],opt.word[1])
        print('The cost is: '+str(edit_distance))
        output_alignment(alignment)
    elif(opt.sentence):
        edit_distance,alignment = sentence_edit_distance(sentence_preprocess(opt.sentence[0]),sentence_preprocess(opt.sentence[1]))
        print('The cost is: '+str(edit_distance))
        output_alignment(alignment)
    elif(opt.batch_word):
        batch_word(opt.batch_word[0],opt.batch_word[1])
    elif(opt.batch_sentence):
        batch_sentence(opt.batch_sentence[0],opt.batch_sentence[1])

    edit_distance, alignment = word_edit_distance("won't", 'wont.')
    # edit_distance, alignment = sentence_edit_distance(sentence_preprocess("i'm victor_mai, my phone is 1223-23422-1 & 1-33"),
    #                                                   sentence_preprocess("victor's name is mai. my phone is 1223-23422-1 and 1-33")
    #                                                   )
    # batch_sentence('sentence_corpus.txt', "sentence_corpus_output.txt")
    # batch_word('word_corpus.txt', "word_corpus_output.txt")
    print("won't")
    print("wont.")
    print('The cost is: '+str(edit_distance))
    output_alignment(alignment)

if __name__ == '__main__':
    import os
    main()
