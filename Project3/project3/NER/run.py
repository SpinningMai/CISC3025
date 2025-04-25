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

def f_beta_score(precision, recall, beta=1.5):
    """beta > 1: recall has higher weight than precision"""
    if precision + recall == 0:
        return 0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

def get_iter_metrics(memm, train_samples, n_iter):
    memm.train(train_samples, n_iter)
    metrics = memm.test()
    beta_f1 = f_beta_score(metrics['precision'], metrics['recall'])
    return beta_f1, copy.deepcopy(memm.classifier)

def ternary_search_best_classifier(memm, left, right) -> (int, MEMM):
    memorized_iter: list[tuple[float, None]] = [(-1, None)] * (right + 1)

    train_samples = memm.extract_samples()

    best_score = -1
    best_iter = -1

    while right - left >= 3:  # Terminate when the range is small enough
        mid1 = left + (right - left) // 3
        mid2 = right - (right - left) // 3

        # Train and evaluate mid1
        if memorized_iter[mid1][0] == -1:
            memorized_iter[mid1] = get_iter_metrics(memm, train_samples, mid1)
        if memorized_iter[mid1][0] > best_score:
            best_score = memorized_iter[mid1][0]
            best_iter = mid1

        if memorized_iter[mid2][0] == -1:
            memorized_iter[mid2] = get_iter_metrics(memm, train_samples, mid2)
        if memorized_iter[mid2][0] > best_score:
            best_score = memorized_iter[mid2][0]
            best_iter = mid2

        # Narrow the search range
        if memorized_iter[mid1][0] < memorized_iter[mid2][0]:
            left = mid1
        else:
            right = mid2

    # Final evaluation (optional: fine-tune in the remaining range)
    return best_iter, memorized_iter[best_iter][1]

def main():
    memm = MEMM("../")

    # Final evaluation (optional: fine-tune in the remaining range)
    # best_iter, memm.best_classifier = ternary_search_best_classifier(memm, left=50, right=60)

    best_iter = 58
    memm.train(memm.extract_samples(), best_iter)
    memm.best_classifier = memm.classifier

    memm.save_model(memm.best_classifier)
    memm.test()
    print('Training finished and best model saved! best iteration is: ', best_iter, '\n')


if __name__ == '__main__':
    #====== Customization ======
    BETA = 0.5
    MAX_ITER = 50
    #==========================

    main()
