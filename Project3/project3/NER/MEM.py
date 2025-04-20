#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# --------------------------------------------------
# Description:
# --------------------------------------------------
# Author: Konfido <konfido.du@outlook.com>
# Created Date : April 4th 2020, 17:45:05
# Last Modified: April 4th 2020, 17:45:05
# --------------------------------------------------

from nltk.classify.maxent import MaxentClassifier
from sklearn.metrics import (accuracy_score, fbeta_score, precision_score,
                             recall_score)
import os
import pickle
import re


class MEMM():
    def __init__(self):
        self.train_path = "data/train"
        self.dev_path = "data/dev"
        self.beta = 0
        self.max_iter = 0
        self.classifier = None
        self.best_classifier = None 
        self.best_f1 = 0  
        self.no_improvement_count = 0 
        self.camel_regex = re.compile(r'^([A-Z]?[a-z]+)+([A-Z][a-z]+)*$')

        self.latin_letters = {'é', 'ü'}
        self.pinyin_regex = re.compile(
            r"^("
            r"(a[io]?|ou?|e[inr]?|ang?|ng|[bmp](a[io]?|[aei]ng?|ei|ie?|ia[no]|o|u)|"
            r"pou|me|m[io]u|[fw](a|[ae]ng?|ei|o|u)|fou|wai|[dt](a[io]?|an|e|[aeio]ng|"
            r"ie?|ia[no]|ou|u[ino]?|uan)|dei|diu|[nl](a[io]?|ei?|[eio]ng|i[eu]?|i?ang?|"
            r"iao|in|ou|u[eo]?|ve?|uan)|nen|lia|lun|[ghk](a[io]?|[ae]ng?|e|ong|ou|u[aino]?|"
            r"uai|uang?)|[gh]ei|[jqx](i(ao?|ang?|e|ng?|ong|u)?|u[en]?|uan)|([csz]h?|"
            r"r)([ae]ng?|ao|e|i|ou|u[ino]?|uan)|[csz](ai?|ong)|[csz]h(ai?|uai|"
            r"uang)|zei|[sz]hua|([cz]h|r)ong|y(ao?|[ai]ng?|e|i|ong|ou|u[en]?|uan))"
            r"){1,4}$"
        )
        self.pinyin_confusion = {"me", "ma", "bin", "fan", "long", "sun", "panda", "china"}

    def features(self, words, previous_label, position):
        """
        Note: The previous label of current word is the only visible label.

        :param words: a list of the words in the entire corpus
        :param previous_label: the label for position-1 (or O if it's the start
                of a new sentence)
        :param position: the word you are adding features for
        """

        features = {}
        """ Baseline Features """
        current_word = words[position]

        # Basic info
        features['has_(%s)' % current_word] = 1
        features['prev_label'] = previous_label

        # Letter cases
        if current_word[0].isupper(): features['Titlecase'] = 1
        if current_word.isupper(): features["Allcapital"] = 1
        if self.camel_regex.fullmatch(current_word):features["Camelcase"] = 1

        # Punctuations
        if "'" in current_word: features["Apostrophe"] = 1
        if "-" in current_word: features["Hyphen"] = 1

        # Suffix
        if current_word.endswith("son"): features["Suffix_son"] = 1
        if current_word.endswith("ez"): features["Suffix_ez"] = 1

        # Prefix
        if current_word.startswith("Mc"): features["Prefix_Mc"] = 1
        if current_word.startswith("O'"): features["Prefix_OAp"] = 1

        # Non-English
        if any(current_word) in self.latin_letters:
            features["Latinletter"] = 1
        if self.pinyin_regex.fullmatch(current_word.lower()):
            features["Pinyin"] = 1
        if current_word.lower().endswith("lyu") or current_word.lower().startswith("lyu"):
            features["Pinyin_lyu"] = 1
        if current_word.lower() in self.pinyin_confusion:
            features["Pinyin_confusion"] = 1

        return features

    def load_data(self, filename):
        words = []
        labels = []
        for line in open(filename, "r", encoding="utf-8"):
            doublet = line.strip().split("\t")
            if len(doublet) < 2:     # remove emtpy lines
                continue
            words.append(doublet[0])
            labels.append(doublet[1])
        return words, labels

    def train(self):
        print('Training classifier...')
        words, labels = self.load_data(self.train_path)
        previous_labels = ["O"] + labels
        features = [self.features(words, previous_labels[i], i)
                    for i in range(len(words))]  # list of dict(str:Any)
        train_samples = [(f, l) for (f, l) in zip(features, labels)]
        
        self.classifier = MaxentClassifier.train(train_samples, max_iter=self.max_iter)

    def test(self):
        print('Testing classifier...')
        words, labels = self.load_data(self.dev_path)
        prev_label = "O"
        results = []
        '''list of str("O" or "PERSON")'''

        for i, word in enumerate(words):
            single_bunch_features = self.features(words, prev_label, i)
            pred_label = self.classifier.classify(single_bunch_features)
            results.append(pred_label)
            prev_label = pred_label  # update the previous label predicted

        f_score = fbeta_score(labels, results, average='macro', beta=self.beta)
        precision = precision_score(labels, results, average='macro')
        recall = recall_score(labels, results, average='macro')
        accuracy = accuracy_score(labels, results)

        print("%-15s %.4f\n%-15s %.4f\n%-15s %.4f\n%-15s %.4f\n" %
              ("f_score=", f_score, "accuracy=", accuracy, "recall=", recall,
               "precision=", precision))

        # 如果当前F1分数更好，则更新最佳模型
        if f_score > self.best_f1:
            self.best_f1 = f_score
            self.best_classifier = self.classifier
            self.no_improvement_count = 0  # reset counter
        else:
            self.no_improvement_count += 1

        # 如果连续三次F1下降，则停止训练
        if self.no_improvement_count >= 3:
            print("Stopping training as F1 score has not improved for 3 consecutive iterations.")
            return False

        return True

    def save_best_model(self):
        if self.best_classifier:
            with open('best_model.pkl', 'wb') as f:
                pickle.dump(self.best_classifier, f)

    def load_model(self):
        with open('best_model.pkl', 'rb') as f:
            self.classifier = pickle.load(f)

    def predict_sentence(self, sentence):
        words = sentence.strip().split()
        predictions = []
        prev_label = "O"
        for i, word in enumerate(words):
            single_bunch_features = self.features(words, prev_label, i)
            pred = self.classifier.classify(single_bunch_features)
            label = "PERSON" if pred[0] > 0.5 else "O"
            predictions.append((word, label))
            prev_label = label
        return predictions


    def show_samples(self, bound):
        """
        Show some sample probability distributions.
        """
        words, labels = self.load_data(self.train_path)
        previous_labels = ["O"] + labels
        features = [self.features(words, previous_labels[i], i)
                    for i in range(len(words))]
        (m, n) = bound
        pdists = self.classifier.prob_classify_many(features[m:n])

        print('  Words          P(PERSON)  P(O)\n' + '-' * 40)
        for (word, label, pdist) in list(zip(words, labels, pdists))[m:n]:
            if label == 'PERSON':
                fmt = '  %-15s *%6.4f   %6.4f'
            else:
                fmt = '  %-15s  %6.4f  *%6.4f'
            print(fmt % (word, pdist.prob('PERSON'), pdist.prob('O')))

    def dump_model(self):
        with open('model.pkl', 'wb') as f:
            pickle.dump(self.classifier, f)

 