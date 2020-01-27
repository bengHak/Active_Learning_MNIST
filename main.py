import os
import glob
import json
import shutil
import argparse

import numpy as np
import tensorflow as tf

from model import Dense

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    # Dataset setting
    parser.add_argument('--learning_rate', type=float, default=0.01, help='-')
    # Optimizer Setting
    parser.add_argument('--batch_size', type=int, default=4000, help='-')
    args, _ = parser.parse_known_args()

    model = Dense(args)
    model.build()

    # 1. 400개 레이블 포함 초기 학습 (labeled)
    model.resetData()
    model.dataLoader.label_manually(400)
    model.train()

    model.resetData()
    model.dataLoader.label_manually(4000)
    model.train()

    model.resetData()
    model.active_learning(manual_label_num=20)

    # model.resetData()
    # model.active_learning(manual_label_num=20)

