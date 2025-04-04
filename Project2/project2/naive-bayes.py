import argparse
import codecs
import heapq
import json
import math
import re
from fractions import Fraction

import numpy as np
from matplotlib import pyplot as plt
from nltk import word_tokenize, PorterStemmer
from sklearn.metrics import f1_score as sklearn_f1_score
from tqdm import tqdm


def preprocess_str(text):
    html_entities = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": "\"",
        "&apos;": "'"
    }
    for entity, replacement in html_entities.items():
        text = text.replace(entity, replacement)

    # Replace them with space
    symbols = r'[?!<>:;\n"]'
    text = re.sub(symbols, " ", text)

    # Replace single quotes with spaces, except "s'"
    text = re.sub(r"(?<!\w)'|'(?!\w|s')", " ", text)

    # Lowercase
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove "^M" signal
    tokens = [re.sub(r'\^m$', '', token) for token in tokens]

    # Remove redundant "."s
    tokens = [re.sub(r'\.{2,}$', '', token) for token in tokens]
    tokens = [re.sub(r'\.$', '', token) for token in tokens]

    # Remove words that do not contain letters or numbers
    tokens = [token for token in tokens if re.search(r'[a-z0-9$]', token)]

    # PorterStemmer
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens


def preprocess(inputfile, outputfile):
    #DONE: preprocess the input file, and output the result to the output file: train.preprocessed.json,test.preprocessed.json
    #   Delete the useless symbols
    #   Convert all letters to the lowercase
    #   Use NLTK.word_tokenize() to tokenize the sentence
    #   Use nltk.PorterStemmer to stem the words
    with open(inputfile, 'r', encoding='utf-8') as f:
        inputdata = json.load(f)
        f.close()

    for each in tqdm(inputdata):
        each[2] = preprocess_str(each[2])

    with codecs.open(outputfile, 'w', encoding='utf-8') as fo:
        fo.write(json.dumps(inputdata))
        fo.close()

    return


def count_word(inputfile, outputfile):
    #DONE: count the words from the corpus, and output the result to the output file in the format required.
    #   A dictionary object may help you with this work.
    with open(inputfile, 'r', encoding='utf-8') as f:
        inputdata = json.load(f)
        f.close()

    to_idx = {'crude': 0, 'grain': 1, 'money-fx': 2, 'acq': 3, 'earn': 4}
    counter = [0] * len(to_idx)

    mydict = {}
    for item in tqdm(inputdata):
        if len(item) < 3:
            continue

        category = item[1]
        words = item[2]

        try:
            idx = to_idx[category]
        except KeyError:
            continue
        counter[idx] += 1

        for word in words:
            if word not in mydict:
                mydict[word] = [0] * len(to_idx)
            mydict[word][idx] += 1

    with open(outputfile, 'w', encoding='utf-8') as fo:
        line = '{} {} {} {} {}\n'.format(*counter)
        fo.write(line)
        for key, item in tqdm(mydict.items()):
            line = '{} {} {} {} {} {}\n'.format(key, *item)
            fo.write(line)
        fo.close()
    return


def feature_selection(inputfile, threshold, outputfile):
    #DONE: Choose the most frequent 10000 words(defined by threshold) as the feature word
    # Use the frequency obtained in 'word_count.txt' to calculate the total word frequency in each class.
    #   Notice that when calculating the word frequency, only words recognized as features are taken into consideration.
    # Output the result to the output file in the format required
    word_count = {}
    with open(inputfile, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            words = line.split(' ')
            if len(words) < 6:
                continue  # First line
            word_count[words[0]] = sum(int(x) for x in words[1:])
        f.close()

    top_keys = heapq.nlargest(min(len(word_count), threshold), word_count.keys(), key=word_count.get)
    top_keys.sort()

    with open(outputfile, 'w', encoding='utf-8') as fo:
        for key in top_keys:
            fo.write('{}\n'.format(key))
        fo.close()
    return


def calculate_probability(word_count, word_dict, outputfile):
    #DONE: Calculate the posterior probability of each feature word, and the prior probability of the class.
    #   Output the result to the output file in the format required
    #   Use 'word_count.txt' and ‘word_dict.txt’ jointly.
    posterior_probability = {}

    with open(word_dict, 'r', encoding='utf-8') as fwd:
        for word in tqdm(fwd):
            word = word.strip()
            posterior_probability[word] = [Fraction(0, 1)] * 5
        fwd.close()

    prior_probability = [Fraction(0)] * 5
    class_word_count = [Fraction(0)] * 5
    one = Fraction(1, 1)
    v = Fraction(len(posterior_probability), 1)

    with open(word_count, 'r', encoding='utf-8') as fwc:
        for line in tqdm(fwc):
            line = line.strip()
            words = line.split(' ')
            if len(words) < 6:  # First line
                for i in range(len(words)):
                    prior_probability[i] = Fraction(int(words[i]))
                n_doc = sum(prior_probability, Fraction(0, 1))
                for i in range(5):
                    prior_probability[i] /= n_doc
            elif words[0] in posterior_probability:
                for i in range(1, len(words)):
                    posterior_probability[words[0]][i - 1] = Fraction(int(words[i]), 1)
                    class_word_count[i - 1] += Fraction(int(words[i]), 1)
            else:
                continue
        fwc.close()

    for key in posterior_probability:
        for i in range(5):
            posterior_probability[key][i] += one
            posterior_probability[key][i] /= (class_word_count[i] + v)
    posterior_probability['<UNKNOWN>'] = [Fraction(1, class_word_count[i] + v + 1)] * 5

    with open(outputfile, 'w', encoding='utf-8') as fo:
        fo.write(' '.join(str(pc) for pc in prior_probability) + '\n')
        for key, item in tqdm(posterior_probability.items()):
            fo.write(key + ' ' + ' '.join(str(post) for post in item) + '\n')
        fo.close()
    return


def classify(probability, testset, outputfile):
    #DONE: Implement the naïve Bayes classifier to assign class labels to the documents in the test set.
    #   Output the result to the output file in the format required
    classes = ['crude', 'grain', 'money-fx', 'acq', 'earn']
    classes_prob = np.zeros(5)  # 使用np.array代替[0.0]*5

    prob = {}
    with open(probability, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            items = line.split(' ')

            if len(items) < 6:
                for i in range(len(items)):
                    frac = Fraction(items[i])
                    classes_prob[i] = math.log(frac.numerator) - math.log(frac.denominator)
                continue

            prob[items[0]] = np.array([math.log(Fraction(item).numerator) - math.log(Fraction(item).denominator)
                                       for item in items[1:6]])

    results = []
    with open(testset, 'r', encoding='utf-8') as f:
        testdata = json.load(f)
        for text in tqdm(testdata):
            final_prob = classes_prob.copy()

            for word in text[2]:
                if word in prob:
                    final_prob += prob[word]
                else:
                    final_prob += prob['<UNKNOWN>']

            predicted_class = classes[final_prob.argmax()]
            results.append((text[0], predicted_class))
        f.close()

    with open(outputfile, 'w', encoding='utf-8') as fo:
        for result in results:
            fo.write('{} {}\n'.format(result[0], result[1]))

    return


def f1_score(testset, classification_result, average='micro'):
    #DONE: Use the F_1 score to assess the performance of the implemented classification model
    #   The return value should be a float object.
    y_true, y_pred = [], []
    with open(testset, 'r', encoding='utf-8') as f:
        testdata = json.load(f)
        for text in tqdm(testdata):
            y_true.append(text[1])
        f.close()

    with open(classification_result, 'r', encoding='utf-8') as f:
        for result in tqdm(f):
            result = result.strip()
            case, predicted_class = result.split(' ')
            y_pred.append(predicted_class)

    micro_average_f1 = sklearn_f1_score(np.array(y_true), np.array(y_pred), average=average)
    return micro_average_f1


def plt_performance(start_n: int = 10, end_n: int = 20000, point_n: int = 50):
    preprocess('train.json', 'train.preprocessed.json')
    preprocess('test.json', 'test.preprocessed.json')
    count_word('train.preprocessed.json', 'word_count.txt')
    num_feature = np.logspace(
        np.log10(start_n),
        np.log10(end_n),
        num=point_n,
        dtype=int
    ).tolist()
    scores = []

    for n in num_feature:
        feature_selection('word_count.txt', n, 'word_dict.txt')
        calculate_probability('word_count.txt', 'word_dict.txt', 'word_probability.txt')
        classify('word_probability.txt', 'test.preprocessed.json', 'classification_result.txt')
        score = f1_score('test.json', 'classification_result.txt')
        scores.append(score)
        print('{} features, F1 score: {:.4f}'.format(n, score))

    plt.figure(figsize=(12, 6))
    plt.plot(num_feature, scores, 'bo-', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.xlabel('Number of features (log scale)')
    plt.ylabel('F1 Score')
    plt.title('Model Performance')
    plt.grid(True, linestyle='--', alpha=0.7)

    max_score = max(scores)
    max_idx = scores.index(max_score)
    plt.annotate(f'Max F1: {max_score:.4f}\n(n={num_feature[max_idx]})',
                 xy=(num_feature[max_idx], max_score),
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->'))

    plt.tight_layout()
    plt.savefig('feature_vs_f1.png', dpi=300)
    plt.show()


def main():
    """ Main Function """

    plt_performance()

    parser = argparse.ArgumentParser()
    parser.add_argument('-pps', '--preprocess', type=str, nargs=2, help='preprocess the dataset')
    parser.add_argument('-cw', '--count_word', type=str, nargs=2, help='count the words from the corpus')
    parser.add_argument('-fs', '--feature_selection', type=str, nargs=3, help='\\select the features from the corpus')
    parser.add_argument('-cp', '--calculate_probability', type=str, nargs=3,
                        help='calculate the posterior probability of each feature word, and the prior probability of the class')
    parser.add_argument('-cl', '--classify', type=str, nargs=3,
                        help='classify the testset documents based on the probability calculated')
    parser.add_argument('-f1', '--f1_score', type=str, nargs=2,
                        help='calculate the F-1 score based on the classification result.')
    parser.add_argument('-run', '--run', type=str, nargs=0, help='run the overall performance plotting.')
    opt = parser.parse_args()

    if opt.preprocess:
        input_file = opt.preprocess[0]
        output_file = opt.preprocess[1]
        preprocess(input_file, output_file)
    elif opt.count_word:
        input_file = opt.count_word[0]
        output_file = opt.count_word[1]
        count_word(input_file, output_file)
    elif opt.feature_selection:
        input_file = opt.feature_selection[0]
        threshold = int(opt.feature_selection[1])
        outputfile = opt.feature_selection[2]
        feature_selection(input_file, threshold, outputfile)
    elif opt.calculate_probability:
        word_count = opt.calculate_probability[0]
        word_dict = opt.calculate_probability[1]
        output_file = opt.calculate_probability[2]
        calculate_probability(word_count, word_dict, output_file)
    elif opt.classify:
        probability = opt.classify[0]
        testset = opt.classify[1]
        outputfile = opt.classify[2]
        classify(probability, testset, outputfile)
    elif opt.f1_score:
        testset = opt.f1_score[0]
        classification_result = opt.f1_score[1]
        f1 = f1_score(testset, classification_result)
        print('The F1 score of the classification result is: ' + str(f1))


if __name__ == '__main__':
    import os

    main()
