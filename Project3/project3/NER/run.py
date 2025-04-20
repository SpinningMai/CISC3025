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

from tqdm import tqdm

from MEM import MEMM


def main():
    classifier = MEMM()

    train_samples = classifier.extract_samples()
    # 开始训练并逐步测试
    for epoch in tqdm(range(MAX_ITER)):
        classifier.train(train_samples, epoch + 2)
        if not classifier.test():
            break  # 如果停止训练，结束循环

    # 保存最好的分类器
    classifier.save_best_model()
    print("Training finished and best model saved!")

    classifier.classifier = classifier.best_classifier
    classifier.test()


if __name__ == '__main__':
    #====== Customization ======
    BETA = 0.5
    MAX_ITER = 500
    #==========================

    main()
