#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chun-Tung Li
"""

import os
import re
import numpy as np
import pandas as pd
from SIMPAD import SSPDetector as ssp

data_folder = 'data/HAPT'
acc_features = ['acc_x', 'acc_y', 'acc_z']

samplingRate = 50

pd.set_option('precision', 2)


def generateGroundTruths(N, _labels, _other_labels):
    groundtruths = [False] * N
    for i in range(len(_labels)):
        start_idx = int(_labels.iloc[i].start_idx)
        stop_idx = int(_labels.iloc[i].stop_idx)
        groundtruths[start_idx:stop_idx] = [True] * (stop_idx - start_idx)
    return groundtruths


def evaluatePerf(_detection, _labels, _other_labels):
    N = len(_detection)
    TP = TN = FP = FN = 0
    mask = np.ones(N, dtype=bool)

    trueN = 0
    for i in range(len(_labels)):
        start_idx = int(_labels.iloc[i].start_idx)
        stop_idx = int(_labels.iloc[i].stop_idx)
        trueN += stop_idx - start_idx + 1
        TP += _detection[start_idx: stop_idx].count(True)
        FN += _detection[start_idx: stop_idx].count(False)
        mask[start_idx: stop_idx] = False

    # Exclusion Zone
    if (len(_other_labels)):
        mask[int(max(_other_labels.stop_idx)): int(min(_labels.start_idx))] = False
    mask[int(max(_labels.stop_idx[_labels.aid == 1])): int(min(_labels.start_idx[_labels.aid > 1]))] = False
    mask = mask.tolist()
    trueN += mask.count(True)

    negRM = np.array(_detection)[mask].tolist()
    TN = negRM.count(False)
    FP = negRM.count(True)

    accuracy = safeDiv((TP + TN), trueN) * 100
    precision = safeDiv(TP, (TP + FP)) * 100
    recall = safeDiv(TP, (TP + FN)) * 100
    F1 = 2 * safeDiv((precision * recall), (precision + recall))

    return [TP, TN, FP, FN, accuracy, precision, recall, F1]


def safeDiv(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 0


# ===================================================
'''
                    Main Program
'''
# ===================================================

# Reading the ground truth
annotation = pd.read_csv(data_folder + '/labels.txt', sep=' ', header=None)
annotation.columns = ['eid', 'uid', 'aid', 'start_idx', 'stop_idx']

annotation['label'] = 'T'  # Transition as default C6
annotation['color'] = 'C2'  # Transition as default C6
annotation.loc[annotation.aid == 1, 'label'] = 'W'  # Walking C0
annotation.loc[annotation.aid == 1, 'color'] = 'C0'
annotation.loc[annotation.aid == 2, 'label'] = 'U'  # Walking Upstairs C1
annotation.loc[annotation.aid == 2, 'color'] = 'C0'
annotation.loc[annotation.aid == 3, 'label'] = 'D'  # Walking Downstairs C2
annotation.loc[annotation.aid == 3, 'color'] = 'C0'
annotation.loc[annotation.aid == 4, 'label'] = 'S'  # Sitting C3
annotation.loc[annotation.aid == 4, 'color'] = 'C1'
annotation.loc[annotation.aid == 5, 'label'] = 'A'  # Standing C4
annotation.loc[annotation.aid == 5, 'color'] = 'C1'
annotation.loc[annotation.aid == 6, 'label'] = 'L'  # Laying C5
annotation.loc[annotation.aid == 6, 'color'] = 'C1'

result = []

# For each raw data
files = os.listdir(data_folder)
for file in files:

    if (not file.startswith('acc')):
        continue

    # Read info from folder name
    format_string = file.split('_')
    sensorType = format_string[0]
    eid = re.search('[0-9]+', format_string[1]).group(0)
    uid = re.search('[0-9]+', format_string[2].replace('.txt', '')).group(0)
    del (format_string)

    # Extract Ground Truths
    labels = annotation[(annotation.eid == int(eid)) & (annotation.uid == int(uid)) & (annotation.aid <= 3)]
    other_labels = annotation[(annotation.eid == int(eid)) & (annotation.uid == int(uid)) & (annotation.aid > 3)]
    if (len(labels) == 0):
        continue

    data = pd.read_csv(data_folder + '/acc_exp' + str(eid) + '_user' + str(uid) + '.txt', names=acc_features, sep=' ', header=None)

    '''
    =================================================================
                                SIMPAD
    =================================================================
    '''

    l = 50
    m = 5 * l

    detection = ssp.SIMPAD(data.T, l, m)
    result.append([eid, uid, 'SIMPAD'] + evaluatePerf(detection, labels, other_labels))

    '''
    =================================================================
                                mSIMPAD
    =================================================================
    '''

    L = np.array([40,50,60])
    m_factor = 5

    detection = ssp.mSIMPAD(data.T, L, m_factor)

    result.append([eid, uid, 'mSIMPAD'] + evaluatePerf(detection, labels, other_labels))

    del (eid, uid, sensorType)
del (file, files)

metrics = ['accuracy', 'precision', 'recall', 'F1']
result_df = pd.DataFrame(result, columns=['eid', 'uid', 'method',
                                          'TP', 'TN', 'FP', 'FN',
                                          'accuracy', 'precision', 'recall', 'F1'])
agg_result = result_df.groupby(['method'])[metrics].agg(['mean', 'std'])
print(agg_result)