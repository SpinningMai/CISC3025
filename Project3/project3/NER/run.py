#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# --------------------------------------------------
# Description:
# --------------------------------------------------
# Author: Du-Haihua <mb75481@um.edu.mo>
# Created Date : April 3rd 2020, 12:05:49
# Last Modified: April 4th 2020, 10:59:35
# --------------------------------------------------

import argparse
import copy

from tqdm import tqdm
from MEM import MEMM

def f_beta_score(precision, recall, beta=2):
    """beta > 1: recall has higher weight than precision"""
    if precision + recall == 0:
        return 0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

def main():
    classifier = MEMM()
    train_samples = classifier.extract_samples()

    left = 0
    right = 50

    best_score = -1
    best_iter = -1
    best_classifier = None

    while right - left >= 3:  # Terminate when the range is small enough
        mid1 = left + (right - left) // 3
        mid2 = right - (right - left) // 3

        # Train and evaluate mid1
        classifier.train(train_samples, mid1 + 2)
        metrics1 = classifier.test()
        beta_f1 = f_beta_score(metrics1['precision'], metrics1['recall'])

        # Train and evaluate mid2
        classifier.train(train_samples, mid2 + 2)
        metrics2 = classifier.test()
        beta_f2 = f_beta_score(metrics2['precision'], metrics2['recall'])

        # Update the best model
        if beta_f1 > best_score:
            best_score = beta_f1
            best_iter = mid1 + 2
            best_classifier = copy.deepcopy(classifier.classifier)
        if beta_f2 > best_score:
            best_score = beta_f2
            best_iter = mid2 + 2
            best_classifier = copy.deepcopy(classifier.classifier)

        # Narrow the search range
        if beta_f1 < beta_f2:
            left = mid1
        else:
            right = mid2

    # Final evaluation (optional: fine-tune in the remaining range)
    classifier.best_classifier = best_classifier
    classifier.save_model(classifier.best_classifier)
    print('Training finished and best model saved! best iteration is: ', best_iter, '\n')


if __name__ == '__main__':
    #====== Customization ======
    BETA = 0.5
    MAX_ITER = 50
    #==========================

    main()
