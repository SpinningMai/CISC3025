#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# --------------------------------------------------
# Description:
# --------------------------------------------------
# Author: Konfido <konfido.du@outlook.com>
# Created Date : April 4th 2020, 17:45:05
# Last Modified: April 4th 2020, 17:45:05
# --------------------------------------------------
import copy
import hashlib
import pickle
import re
import string
import os

import nltk
import unicodedata
from nltk import word_tokenize
from nltk.classify.maxent import MaxentClassifier
from nltk.corpus import stopwords
from sklearn.metrics import (accuracy_score, fbeta_score, precision_score, recall_score)

class MEMM:
    def __init__(self, prev_path):
        self.train_path = os.path.join(prev_path, "data/train")
        self.dev_path = os.path.join(prev_path, "data/dev")
        self.first_name_path = os.path.join(prev_path, "data/first_names.all.txt")
        self.cn_last_name_path = os.path.join(prev_path, "data/cn_last_names.all.txt")
        self.cn_first_name_path = os.path.join(prev_path, "data/cn_first_names.all.txt")
        self.last_name_path = os.path.join(prev_path, "data/last_names.all.txt")
        self.model_path = os.path.join(prev_path, "model.pkl")

        self.beta = 0
        self.classifier = None
        self.best_classifier = None
        self.camel_regex = re.compile(r'^([a-z]+([A-Z]+[a-z]*)+)|([A-Z]+[a-z]*){2,}$')
        self.titles = {"mr", "mrs", "ms", "dr", "prof", "rev", "sir", "madam", "miss"}
        nltk.download('stopwords')
        self.nltk_stopwords = set(stopwords.words('english'))

        def load_gazetteer(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return {word.strip() for word in f if word.strip()}
            except FileNotFoundError:
                print(f"Warning: File {file_path} not found, returning empty set")
                return set()
            except UnicodeDecodeError:
                print(f"Error: Failed to decode {file_path} as UTF-8")
                return set()

        self.first_name_gazetteer = load_gazetteer(self.first_name_path)
        self.last_name_gazetteer = load_gazetteer(self.last_name_path)
        self.cn_first_name_gazetteer = load_gazetteer(self.cn_first_name_path)
        self.cn_last_name_gazetteer = load_gazetteer(self.cn_last_name_path)

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

    @staticmethod
    def is_latin_char(check_char):
        try:
            return unicodedata.name(check_char).startswith('LATIN')
        except ValueError:
            return False


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
        current_word_lower = current_word.lower()

        # Roughly features of current word
        features['has_(%s)' % current_word] = 1
        features['cur_word_len'] = len(current_word) // 2
        if current_word_lower in self.nltk_stopwords: features['cur_word_is_stopword'] = 1

        # Gazetteer Checking
        features['cur_word_in_first_name_gazetteer'] = (current_word_lower in self.first_name_gazetteer)
        features['cur_word_in_last_name_gazetteer'] = (current_word_lower in self.last_name_gazetteer)
        features['cur_word_in_cn_first_name_gazetteer'] = (current_word_lower in self.cn_first_name_gazetteer)
        features['cur_word_in_cn_last_name_gazetteer'] = (current_word_lower in self.cn_last_name_gazetteer)

        # Roughly features of previous word
        prev_word = words[position - 1] if position > 0 else "."

        features['prev_word_label'] = previous_label
        features['prev_word_is_capitalized'] = prev_word[0].isupper()
        features['prev_word_ends_with_punctuation'] = prev_word[-1] in string.punctuation
        features['prev_word_is_title'] = any((prev_word.lower().startswith(t) and
                                              len(prev_word.lower()) <= len(t) + 1)
                                             for t in self.titles) # Mr, Mr., Miss., ...
        features['prev_word_len'] = len(prev_word) // 2
        features['prev_word_is_digit'] = prev_word.isdigit()

        # Letter cases
        if current_word[0].isupper(): features['Titlecase'] = 1
        if current_word.isupper(): features["Allcapital"] = 1
        if self.camel_regex.fullmatch(current_word): features["Camelcase"] = 1

        # Punctuations
        if "'" in current_word: features["Apostrophe"] = 1
        if "-" in current_word: features["Hyphen"] = 1

        # Prefix & Suffix hashing
        max_record_len = 4

        _prefix = current_word[:max_record_len]
        _suffix = current_word[-max_record_len:]
        features["prefix_hash"] = int(hashlib.md5(_prefix.encode()).hexdigest(), 16) % 100000
        features["suffix_hash"] = int(hashlib.md5(_suffix.encode()).hexdigest(), 16) % 100000

        # Non-English
        if self.pinyin_regex.fullmatch(current_word_lower):
            features["Pinyin"] = 1
        if current_word_lower.endswith("lyu") or current_word_lower.startswith("lyu"):
            features["Pinyin_lyu"] = 1 # specialize for 吕
        if current_word_lower in self.pinyin_confusion:
            features["Pinyin_confusion"] = 1

        if any(ord(c) > 127 and self.is_latin_char(c) for c in current_word):
            features["Contain_non_ascii_latin"] = 1
        if any(char.isdigit() for char in current_word):
            features["Contain_any_number"] = 1
        if not current_word.isalpha():
            features["Contain_no_alpha"] = 1

        return features

    @staticmethod
    def load_data(filename):
        words = []
        labels = []
        for line in open(filename, "r", encoding="utf-8"):
            doublet = line.strip().split("\t")
            if len(doublet) < 2:     # remove emtpy lines
                continue
            words.append(doublet[0])
            labels.append(doublet[1])
        return words, labels

    def extract_samples(self):
        words, labels = self.load_data(self.train_path)
        previous_labels = ["O"] + labels
        features = [self.features(words, previous_labels[i], i)
                    for i in range(len(words))]  # list of dict(str:Any)
        train_samples = [(f, l) for (f, l) in zip(features, labels)]

        return train_samples

    def train(self, train_samples, max_iter:int) -> None:
        self.classifier = MaxentClassifier.train(train_samples, max_iter=max_iter)

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

        print("\n%-15s %.4f\n%-15s %.4f\n%-15s %.4f\n%-15s %.4f\n" %
              ("f_score=", f_score, "accuracy=", accuracy, "recall=", recall,
               "precision=", precision))

        return {'f_score': f_score, 'accuracy': accuracy, 'precision': precision, 'recall': recall}

    def save_model(self, classifier_to_save):
        if classifier_to_save is not None:
            with open(self.model_path, 'wb') as f:
                pickle.dump(classifier_to_save, f)

    def load_model(self):
        with open(self.model_path, 'rb') as f:
            self.classifier = pickle.load(f)

    def predict_sentence(self, sentence):
        """
        使用 MEMM 模型进行预测。
        :param sentence: 输入的句子
        """
        words = word_tokenize(sentence)
        predictions = []
        previous_label = "O"

        for i in range(len(words)):
            single_bunch_features = self.features(words, previous_label, i)

            current_label = self.classifier.classify(single_bunch_features)
            predictions.append(current_label)

            previous_label = current_label

        return list(zip(words, predictions))


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

 