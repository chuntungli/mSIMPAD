#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chun-Tung Li
"""

import os
import numpy as np
import pandas as pd
from SIMPAD import SSPDetector as ssp

data_folder = 'data/PAMAP2'
acc_features = ['acc_x', 'acc_y', 'acc_z']

samplingRate = 100
newSamplingRate = 50
transRatio = newSamplingRate / samplingRate

repeatActs = [4, 5, 6, 7, 12, 13, 24]
nonRepeatActs = [1, 2, 3, 8, 9, 10]

pd.set_option('precision', 2)

def safeDiv(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 0


def evalPerf(_detection, _label):
    _detection = np.array(_detection)
    _label = np.array(_label)

    if (_detection.shape[0] != _label.shape[0]):
        print('Length not match between detection and label.')

    TP = np.sum(_detection & _label)
    TN = np.sum(~_detection & ~_label)
    FP = np.sum(_detection & ~_label)
    FN = np.sum(~_detection & _label)
    errRate = safeDiv(FP + FN, len(_detection)) * 100
    accuracy = safeDiv(TP + TN, len(_detection)) * 100
    precision = safeDiv(TP, TP + FP) * 100
    recall = safeDiv(TP, TP + FN) * 100
    F1 = 2 * safeDiv((precision * recall), (precision + recall))

    return [TP, TN, FP, FN, errRate, accuracy, precision, recall, F1]


# ===================================================
'''
                    Main Program
'''
# ===================================================


result = []

files = os.listdir(data_folder)
for file in files:
    if not file.endswith('.dat'):
        continue

    # Reading From Raw Data
    data = pd.read_csv('%s/%s' % (data_folder, file), sep='\s+', header=None)
    data = data.iloc[:, [0, 1, 38, 39, 40]]  # Using Ankle Accelerometers
    data.columns = ['time', 'activityID'] + acc_features
    data.index = pd.to_datetime(data.time, unit='s')
    data = data.drop('time', 1)

    # Resample Data
    data = data.resample("%.7fS" % (1 / newSamplingRate)).mean()

    # Interpolate missing data
    data = data.interpolate()

    data.index = range(len(data))

    # Generate Labels
    data.activityID = data.activityID.astype(int)
    data['label'] = data.activityID.isin(repeatActs)

    '''
    =================================================================
                                SIMPAD
    =================================================================
    '''

    # Optimal Parameters
    l = 50
    m = 5 * l

    detection = ssp.SIMPAD(data[acc_features].T, l, m)

    N = len(detection)
    mask = (data.activityID > 0)[:N]
    detection_excluded = np.array(detection)[mask]
    label_excluded = data.label[:N][mask]
    result.append(['SIMPAD', file] + evalPerf(detection_excluded, label_excluded))

    '''
    =================================================================
                                mSIMPAD
    =================================================================
    '''

    # Optimal Parameters
    L = np.array([40,70,100])
    m_factor = 5

    detection = ssp.mSIMPAD(data[acc_features].T, L, m_factor)

    N = len(detection)
    mask = (data.activityID > 0)[:N]
    detection_excluded = np.array(detection)[mask]
    label_excluded = data.label[:N][mask]
    result.append(['mSIMPAD', file] + evalPerf(detection_excluded, label_excluded))

result_df = pd.DataFrame(result, columns=['method', 'file',
                                          'TP', 'TN', 'FP', 'FN',
                                          'errRate', 'accuracy', 'precision', 'recall', 'F1'])
aggResult = result_df.groupby(['method'])[['accuracy', 'precision', 'recall', 'F1']].agg(['mean', 'std'])
print(aggResult)
