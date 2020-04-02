#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chun-Tung Li
"""

import os
import re
import numpy as np
import pandas as pd

from SIMPAD.RCMP import rcmstomp
from parseTraces_ubicomp import parseTrace

data_folder = 'data/ubicomp13 walking'

# Read the annotation
annotation = []
for line in open(data_folder + '/groundtruth_WD.txt', 'r'):
    if (re.match('p[0-9]+.[0-9]+', line) is None):
        continue

    format_string = re.search('p[0-9]+.[0-9]+', line).group(0)
    line = line.replace(format_string, '')
    format_string = format_string.split('.')
    uid = format_string[0]
    eid = format_string[1]

    format_string = line.split('     ')
    placement = format_string[0]
    placement = placement.replace('_', ' ')

    indices = re.findall('[0-9]+', format_string[1])
    start_idx = int(indices[0])
    end_idx = int(indices[1])

    annotation.append([uid, eid, placement, start_idx, end_idx])
del (line, format_string, uid, eid, placement, indices, start_idx, end_idx)
annotation = pd.DataFrame(annotation, columns=['uid', 'eid', 'placement', 'start_idx', 'stop_idx'])

files = os.listdir(data_folder)
result = []
for file in files:
    if (not file.endswith('dat') and not file.endswith('out')):
        continue

    format_string = file.split('.')
    uid = format_string[0]
    format_string = format_string[1].split('_')
    eid = format_string[0]
    gender = format_string[1]
    age = format_string[2]
    height = format_string[3]
    placement = ' '.join(format_string[4:])

    ground_truth = annotation.loc[
        (annotation.uid == uid) & (annotation.eid == eid) & (annotation.placement == placement)]

    if (ground_truth.shape[0] == 0):
        continue

    (accTs, accData, gyroTs, gyroData, magnTs, magnData) = parseTrace(data_folder + '/' + file)
    data = np.array(accData).T

    l = 100
    m = 6 * l
    tau = 0.85
    e = 100

    mp, ip = rcmstomp(data, l, m)

    # Calculate rolling mean of MP
    ma_MP = pd.Series(mp[2, :])
    ma_MP = ma_MP.rolling(l).mean()

    # Determine threshold
    threshold = np.nanmean(ma_MP[:e]) * tau

    RM = [False] * len(ma_MP)

    counter = 0
    for i in range(0, len(ma_MP)):
        if (ma_MP[i] <= threshold):
            if (counter >= l):
                RM[i - l:i] = [True] * l
            else:
                counter += 1
        else:
            counter = 0
    del (counter, i)

    TP = TN = FP = FN = 0
    TP = RM[int(ground_truth.start_idx):int(ground_truth.stop_idx)].count(True)
    FP = RM[e:int(ground_truth.start_idx)].count(True) + RM[int(ground_truth.stop_idx):].count(True)
    TN = RM[e:int(ground_truth.start_idx)].count(False) + RM[int(ground_truth.stop_idx):].count(False)
    FN = RM[int(ground_truth.start_idx):int(ground_truth.stop_idx)].count(False)

    result.append([eid, uid, len(RM), TP, TN, FP, FN, TP / (TP + FN),
                     TN / (TN + FP), FP / (TN + FP), FN / (TP + FN), (FP + FN) / len(RM)])

result_df = pd.DataFrame(result, columns=['eid', 'uid', 'N', 'TP', 'TN', 'FP', 'FN',
                                          'TPR', 'TNR', 'FPR', 'FNR', 'errRate'])

print(result_df[['TPR', 'TNR', 'FPR', 'FNR', 'errRate']].agg('median'))
print('\nMedian of Error Rate: %.2f%%' % (result_df.aggregate('errRate').median() * 100))