#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chun-Tung Li
"""

import timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SIMPAD.RCMP import rcmstomp
from SIMPAD.RCMP import rcmstamp

from mSTOMP.mstomp import mstomp
from mSTOMP.mstamp import mstamp
from mSTOMP.scrimp import scrimp_plus_plus as scrimp

Fs = 100

# ============================================================================
#                         Comparison Among Algorithms
# ============================================================================

sub_len = 100
sea_ran = 2 * sub_len

result = []
for length in range(10, 18):
    for trail in range(2):
        # Generate sequence at 2^length
        data = np.sin(2 * np.pi * np.arange(2 ** length) / Fs)
        seq_len = data.shape[0]

        # Time RC-mSTOMP
        print('Running RC-mSTOMP for sequence length %d' % seq_len)
        start_time = timeit.default_timer()
        mp, ip = rcmstomp(data, sub_len, sea_ran)
        rcmstomp_time = timeit.default_timer() - start_time
        print('\nRC-mSTOMP elapse time for subsequence length %d is %.2f (secs)' % (seq_len, rcmstomp_time))

        # # Time RC-mSTAMP
        # print('Running RC-mSTAMP for sequence length %d' % seq_len)
        # start_time = timeit.default_timer()
        # mp, ip = rcmstamp(data, sub_len, sea_ran)
        # rcmstamp_time = timeit.default_timer() - start_time
        # print('\nRC-mSTAMP elapse time for subsequence length %d is %.2f (secs)' % (seq_len, rcmstamp_time))

        # Time mSTOMP
        print('Running mSTOMP for sequence length %d' % seq_len)
        start_time = timeit.default_timer()
        mp, ip = mstomp(data, sub_len)
        mstomp_time = timeit.default_timer() - start_time
        print('\nmSTOMP elapse time for subsequence length %d is %.2f (secs)' % (seq_len, mstomp_time))

        # # Time mSTAMP
        # print('Running mSTAMP for sequence length %d' % seq_len)
        # start_time = timeit.default_timer()
        # mp, ip = mstamp(data, sub_len)
        # mstamp_time = timeit.default_timer() - start_time
        # print('\nmSTAMP elapse time for subsequence length %d is %.2f (secs)' % (seq_len, mstomp_time))

        # Time SCRIMP
        print('Running SCRIMP++ for sequence length %d' % seq_len)
        start_time = timeit.default_timer()
        mp, ip = scrimp(data, sub_len, random_state=42)
        scrimp_time = timeit.default_timer() - start_time
        print('\nSCRIMP++ elapse time for subsequence length %d is %.2f (secs)' % (seq_len, scrimp_time))

        result.append([seq_len, trail, rcmstomp_time, mstomp_time, scrimp_time])

result_df = pd.DataFrame(result, columns=['length', 'trail', 'rcmstomp', 'mstomp', 'scrimp'])
aggResult = result_df.groupby('length')[['rcmstomp', 'mstomp', 'scrimp']].mean()

fig = plt.figure(figsize = (5,3.5), dpi = 120)
plt.plot(aggResult.index, aggResult.rcmstomp, linewidth=1,
         marker = 's', fillstyle = 'none', label = 'RC-mSTOMP', color='b')
# plt.plot(aggResult.index, aggResult.rcmstamp, linewidth=1,
#          marker = 'x', fillstyle = 'none', label = 'RC-mSTAMP')
plt.plot(aggResult.index, aggResult.mstomp, linewidth=1,
         marker = '^', fillstyle = 'none', label = 'mSTOMP', color='r')
# plt.plot(aggResult.index, aggResult.mstamp, linewidth=1,
#          linestyle = '-.', marker = 'o', fillstyle = 'none', label = 'mSTAMP')
plt.plot(aggResult.index, aggResult.scrimp, linewidth=1,
         marker = 'o', fillstyle = 'none', label = 'SCRIMP++', color='g')
plt.xticks(np.arange(0, 140000, step=20000), ['0', '20k', '40k', '60k', '80k', '100k', '120k'])
plt.xlabel('Sequence Length', fontsize=12)
plt.ylabel('Time (sec)', fontsize=12)
plt.gca().yaxis.grid(True, linestyle='--')
plt.legend(loc='upper left', prop={'size': 12})
fig.savefig('fig_speed_compare.pdf', format='pdf', bbox_inches='tight')
plt.show()
plt.close(fig)

del (sub_len, sea_ran, result, result_df, aggResult)

# ============================================================================
#                    Time Complexity of Sequence Length n
# ============================================================================

sub_len = 100
sea_ran = 2 * sub_len

result = []
for length in range(10, 22):
    data = np.sin(2 * np.pi * np.arange(2 ** length) / Fs)
    seq_len = data.shape[0]
    for trail in range(2):
        print('Running RC-mSTOMP for sequence length %d' % seq_len)
        start_time = timeit.default_timer()
        mp, ip = rcmstomp(data, sub_len, sea_ran)
        rcmstomp_time = timeit.default_timer() - start_time
        print('\ncmSTOMP elapse time for sequence length %d is %.2f sec' % (seq_len, rcmstomp_time))
        result.append([seq_len, trail, rcmstomp_time])

result_df = pd.DataFrame(result, columns=['length', 'trail', 'time'])
aggResult_n = result_df.groupby('length')[['time']].mean()

del (sub_len, sea_ran, result, result_df)

# ============================================================================
#                     Time Complexity of Motif Length l
# ============================================================================

length = 14 # 2^14
sea_ran = 1000

result = []
for sub_len in range(50, 510, 50):
    data = np.sin(2 * np.pi * np.arange(2 ** length) / Fs)
    for trail in range(2):
        print('Running RC-mSTOMP for subsequence length ' + str(sub_len))
        start_time = timeit.default_timer()
        mp, ip = rcmstomp(data, sub_len, sea_ran)
        rcmstomp_time = timeit.default_timer() - start_time
        print('\ncmSTOMP elapse time for subsequence length ' + str(length) + ' is ' + str(rcmstomp_time) + ' sec')
        result.append([sub_len, trail, rcmstomp_time])

result_df = pd.DataFrame(result, columns=['sub_len', 'trail', 'time'])
aggResult_l = result_df.groupby('sub_len')[['time']].mean()

del (length, sub_len, sea_ran, result, result_df)

# ============================================================================
#                   Time Complexity of search range m
# ============================================================================

length = 16 # 2^16
sub_len = 500

result = []
for sea_ran in range(5000, 55000, 5000):
    data = np.sin(2 * np.pi * np.arange(2 ** length) / Fs)
    for trail in range(2):
        print('Running RC-mSTOMP for searching range %d' % sea_ran)
        start_time = timeit.default_timer()
        mp, ip = rcmstomp(data, sub_len, sea_ran)
        rcmstomp_time = timeit.default_timer() - start_time
        print('\nrcmSTOMP elapse time for searching range %d is %.2f sec.' % (sea_ran, rcmstomp_time))
        result.append([sea_ran, trail, rcmstomp_time])

result_df = pd.DataFrame(result, columns=['sea_ran', 'trail', 'time'])
aggResult_m = result_df.groupby('sea_ran')[['time']].mean()

del (length, sub_len, sea_ran, result_df)

# ============================================================================
#                     Time Complexity of Dimensionality d
# ============================================================================

length = 14
sub_len = 100
sea_ran = 2 * sub_len

result = []
for d in range(5, 251, 5):
    data = np.array([list(np.sin(2 * np.pi * np.arange(2 ** length) / Fs)) for i in range(d)])
    for trail in range(2):
        print('Running RC-mSTOMP for number of dimensions %d' % d)
        start_time = timeit.default_timer()
        mp, ip = rcmstomp(data, sub_len, sea_ran)
        rcmstomp_time = timeit.default_timer() - start_time
        print('\ncmSTOMP elapse time for number of dimensions %d is %.2f sec' % (d, rcmstomp_time))
        result.append([d, trail, rcmstomp_time])

result_df = pd.DataFrame(result, columns=['dimension', 'trail', 'time'])
aggResult_d = result_df.groupby('dimension')[['time']].mean()

del (length, sub_len, sea_ran, result_df)

# Generate Plots

fig, axes = plt.subplots(1,4, figsize=(12,3), dpi=120)

axes[0].plot(aggResult_n.index, aggResult_n.time, color = 'r')
axes[0].set_xticks(np.arange(0, 2100000, step=500000))
axes[0].set_xticklabels(['0', '0.5m', '1m', '1.5m', '2m'])
axes[0].set_xlim((-100000, 2200000))
axes[0].set_xlabel('Time Series Length', fontsize=12)

axes[1].plot(aggResult_l.index, aggResult_l.time, color = 'r')
axes[1].set_xticks(np.arange(100, 600, step=100))
axes[1].set_xlim((40, 510))
axes[1].set_ylim((0, 6))
axes[1].set_xlabel('Subsequence Length', fontsize=12)

axes[2].plot(aggResult_m.index, aggResult_m.time, color = 'r')
axes[2].set_xticks(np.arange(10000, 60000, step=10000))
axes[2].set_xticklabels(['10k', '20k', '30k', '40k', '50k'])
axes[2].set_xlabel('Searching Range', fontsize=12)

axes[3].plot(aggResult_d.index, aggResult_d.time, color = 'r')
axes[2].set_xticks(np.arange(0, 300, step=50))
axes[2].set_xlim((-5, 255))
axes[2].set_xlabel('Dimension', fontsize=12)

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.ylabel("Time (sec)", fontsize=12, labelpad=10)
fig.tight_layout()
fig.savefig('fig_speed_all.pdf', format='pdf', bbox_inches='tight')
plt.show()
plt.close(fig)