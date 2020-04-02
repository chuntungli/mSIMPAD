#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chun-Tung Li
"""

import numpy as np
import matplotlib.pyplot as plt
from SIMPAD import SSPDetector as ssp

np.random.seed(42)

seq_len = 1000
seq = np.zeros(seq_len)

# Insert Sine Wave of length 15
st_idx = 100
ed_idx = 300
pattern_length = 15
seq[st_idx:ed_idx] = np.sin(2 * np.pi * np.arange(ed_idx - st_idx) / pattern_length)

# Insert Sine Wave of length 25
st_idx = 400
ed_idx = 600
pattern_length = 25
seq[st_idx:ed_idx] = np.sin(2 * np.pi * np.arange(ed_idx - st_idx) / pattern_length)

# Insert Sine Wave of length 40
st_idx = 700
ed_idx = 900
pattern_length = 40
seq[st_idx:ed_idx] = np.sin(2 * np.pi * np.arange(ed_idx - st_idx) / pattern_length)

# Insert Random Noise
seq = seq + np.random.normal(0, 0.2, seq_len)

# Detection with SIMPAD
detection = ssp.SIMPAD(seq, l=30, m=60)
fig, axes = plt.subplots(2, 1, figsize=(12,3), dpi=300, sharex=True)
axes[0].plot(seq)
axes[1].plot(detection)
plt.show()
plt.close(fig)

# Detection with mSIMPAD
detection = ssp.mSIMPAD(seq, L=[15,30,40], m_factor=2)
fig, axes = plt.subplots(2, 1, figsize=(12,3), dpi=300, sharex=True)
axes[0].plot(seq)
axes[1].plot(detection)
plt.show()
plt.close(fig)