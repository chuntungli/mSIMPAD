#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Range-Constrained multidimensional matrix profile with RC-mSTOMP (stomp based)

Parameters
----------
seq(T) : numpy matrix, shape (n_dim, seq_len)
    input sequence
sub_len(l) : int
    subsequence length
sea_ran(m) : int
    search range
return_dimension : bool
    if True, also return the matrix profile dimension. It takses O(d^2 n)
    to store and O(d^2 n^2) to compute. (default is False)

Returns
-------
matrix_profile : numpy matrix, shape (n_dim, sub_num)
    matrix profile
profile_index : numpy matrix, shape (n_dim, sub_num)
    matrix profile index
profile_dimension : list, optional, shape (n_dim)
    matrix profile dimension, this is only returned when return_dimension
    is True

Notes
-----
This code is modified from the code provided in the following URL
https://sites.google.com/view/mstamp/

@author: Chun-Tung Li
"""

from __future__ import print_function
import time
import numpy as np

_EPS = 1e-14

def _mass_pre(seq, sub_len):
    seq_len = len(seq)
    seq_cum = np.cumsum(seq)
    seq_sq_cum = np.cumsum(np.square(seq))
    seq_sum = (seq_cum[sub_len - 1:seq_len] -
               np.concatenate(([0], seq_cum[0:seq_len - sub_len])))
    seq_sq_sum = (seq_sq_cum[sub_len - 1:seq_len] -
                  np.concatenate(([0], seq_sq_cum[0:seq_len - sub_len])))
    seq_mu = seq_sum / sub_len
    seq_sig_sq = seq_sq_sum / sub_len - np.square(seq_mu)
    seq_sig = np.sqrt(seq_sig_sq)
    return seq_mu, seq_sig


def _mass(seq_seg, que, sub_len, mu_seg, sig_seg):
    seg_len = len(seq_seg)
    seq_pad = np.zeros(2 * seg_len)
    seq_pad[0:seg_len] = seq_seg
    seq_freq = np.fft.fft(seq_pad)

    que = que[::-1]  # Reverse order of que
    que_pad = np.zeros(2 * seg_len)
    que_pad[0:sub_len] = que
    que_freq = np.fft.fft(que_pad)

    product_freq = seq_freq * que_freq
    product = np.fft.ifft(product_freq)
    product = np.real(product)

    que_sum = np.sum(que)
    que_sq_sum = np.sum(np.square(que))
    que_mu = que_sum / sub_len
    que_sig_sq = que_sq_sum / sub_len - que_mu ** 2
    if que_sig_sq < _EPS:
        que_sig_sq = _EPS
    que_sig = np.sqrt(que_sig_sq)

    dist_profile = (2 * (sub_len - (product[sub_len - 1:seg_len] -
                                    sub_len * mu_seg * que_mu) /
                         (sig_seg * que_sig)))
    return dist_profile, que_sig

def _calMuSig(seq):
    seq_len = len(seq)
    seq_sum = np.sum(seq)
    seq_sq_sum = np.sum(np.square(seq))
    seq_mu = seq_sum / seq_len
    seq_sig_sq = seq_sq_sum / seq_len - seq_mu ** 2
    if seq_sig_sq < _EPS:
        seq_sig_sq = _EPS
    seq_sig = np.sqrt(seq_sig_sq)
    return seq_sum, seq_sq_sum, seq_sig


def _dotProduct(seq, que):
    seq_len = len(seq)
    sub_len = len(que)

    seq_pad = np.zeros(seq_len + sub_len - 1)
    seq_pad[0:seq_len] = seq
    seq_freq = np.fft.fft(seq_pad)

    que = que[::-1]
    que_pad = np.zeros(seq_len + sub_len - 1)
    que_pad[0:sub_len] = que
    que_freq = np.fft.fft(que_pad)

    product_freq = seq_freq * que_freq
    product = np.fft.ifft(product_freq)
    product = np.real(product)

    return product[sub_len - 1:seq_len]


def rcmstomp(seq, sub_len, sea_ran, return_dimension=False):
    if sub_len < 4:
        raise RuntimeError('Subsequence length (sub_len) must be at least 4')
    # if sea_ran / sub_len < 2:
    #     raise RuntimeError('Search range (sea_ran) must be at least 2x subsequence length (sub_len))')

    exc_zone = sub_len // 2
    seq = np.array(seq, dtype=float, copy=True)

    if seq.ndim == 1:
        seq = np.expand_dims(seq, axis=0)

    seq_len = seq.shape[1]  # n
    sub_num = seq.shape[1] - sub_len + 1  # n-l+1 (nDP)
    n_dim = seq.shape[0]  # d
    skip_loc = np.zeros(sub_num, dtype=bool)
    for i in range(sub_num):
        if not np.all(np.isfinite(seq[:, i:i + sub_len])):
            skip_loc[i] = True
    seq[~np.isfinite(seq)] = 0

    seq_mu = np.empty((n_dim, sub_num))
    seq_sig = np.empty((n_dim, sub_num))
    for i in range(n_dim):
        seq_mu[i, :], seq_sig[i, :] = _mass_pre(seq[i, :], sub_len)

    if return_dimension:
        profile_dimension = []
        for i in range(n_dim):
            profile_dimension.append(np.empty((i + 1, sub_num), dtype=int))

    matrix_profile = np.empty((n_dim, sub_num))
    matrix_profile[:] = np.inf
    profile_index = -np.ones((n_dim, sub_num), dtype=int)

    last_product = np.empty((n_dim, 2 * sea_ran - sub_len + 2))
    first_product = np.empty((n_dim, sea_ran + 1))
    last_product[:] = first_product[:] = np.inf

    drop_val = np.empty(n_dim)

    que_sum = np.empty(n_dim)
    que_sq_sum = np.empty(n_dim)
    que_sig = np.empty(n_dim)

    tic = time.time()
    for i in range(sub_num):
        cur_prog = (i + 1) / sub_num
        time_left = ((time.time() - tic) / (i + 1)) * (sub_num - i - 1)
        print('\rProgress [{0:<50s}] {1:5.1f}% {2:8.1f} sec'
              .format('#' * int(cur_prog * 50),
                      cur_prog * 100, time_left), end="")

        win_st = max(0, i - sea_ran)
        win_ed = min(sub_num, i + sea_ran - sub_len + 2)

        trans_i = i
        trans_st = max(0, sea_ran - i)
        trans_ed = min(2 * sea_ran - sub_len + 2,
                       2 * sea_ran - sub_len + 2 - (i + sea_ran - seq_len + 1))

        dist_profile = np.empty((n_dim, trans_ed - trans_st))
        dist_profile[:] = np.inf

        for j in range(n_dim):
            que = seq[j, i:i + sub_len]

            mu_seg = seq_mu[j, win_st: win_ed]
            sig_seg = seq_sig[j, win_st: win_ed]

            if i == 0:
                (que_sum[j], que_sq_sum[j], que_sig[j]) = _calMuSig(que)
                que_mu = que_sum[j] / sub_len

                first_product[j, :] = _dotProduct(seq[j, : sea_ran + sub_len], que)
                last_product[j, sea_ran:] = first_product[j, : -(sub_len - 1)].copy()

                dist_profile[j, :] = (2 * (sub_len -
                                           (last_product[j, sea_ran:] - sub_len * mu_seg * que_mu) /
                                           (sig_seg * que_sig[j])))
            else:
                que_sum[j] = que_sum[j] - drop_val[j] + que[-1]
                que_sq_sum[j] = que_sq_sum[j] - drop_val[j] ** 2 + que[-1] ** 2
                que_mu = que_sum[j] / sub_len
                que_sig_sq = que_sq_sum[j] / sub_len - que_mu ** 2
                if que_sig_sq < _EPS:
                    que_sig_sq = _EPS
                que_sig[j] = np.sqrt(que_sig_sq)

                last_product[j, :trans_st] = np.inf
                last_product[j, trans_ed:] = np.inf

                if i <= sea_ran:
                    last_product[j, trans_st + 1: trans_ed] = (last_product[j, trans_st + 1: trans_ed] -
                                                               (drop_val[j] * seq[j, win_st: win_ed - 1]) +
                                                               (que[-1] * seq[j,
                                                                          win_st + sub_len: win_ed + sub_len - 1]))
                    last_product[j, trans_st] = first_product[j, i]
                else:
                    last_product[j, trans_st: trans_ed] = (last_product[j, trans_st: trans_ed] -
                                                           (drop_val[j] * seq[j, win_st - 1: win_ed - 1]) +
                                                           que[-1] * seq[j, win_st + sub_len - 1: win_ed + sub_len - 1])
                    trans_i = sea_ran

                dist_profile[j, :] = (2 * (sub_len -
                                           (last_product[j, trans_st: trans_ed] - sub_len * mu_seg * que_mu) /
                                           (sig_seg * que_sig[j])))

            drop_val[j] = que[0]
            dist_profile[dist_profile < _EPS] = 0

        if skip_loc[i] or np.any(que_sig < _EPS):
            continue

        exc_zone_st = max(0, trans_i - exc_zone)
        exc_zone_ed = min(dist_profile.shape[1], trans_i + exc_zone + 1)
        dist_profile[:, exc_zone_st:exc_zone_ed] = np.inf
        dist_profile[:, skip_loc[win_st: win_ed]] = np.inf
        dist_profile[seq_sig[:, win_st: win_ed] < _EPS] = np.inf

        dist_profile_dim = np.argsort(dist_profile, axis=0)
        dist_profile_sort = np.sort(dist_profile, axis=0)
        dist_profile_cumsum = np.zeros(dist_profile.shape[1])
        for j in range(n_dim):
            dist_profile_cumsum += dist_profile_sort[j, :]
            dist_profile_mean = dist_profile_cumsum / (j + 1)
            update_pos = dist_profile_mean < matrix_profile[j, win_st: win_ed]
            profile_index[j, win_st: win_ed][update_pos] = i
            matrix_profile[j, win_st: win_ed][update_pos] = dist_profile_mean[update_pos]
            if return_dimension:
                profile_dimension[j][:, update_pos] = \
                    dist_profile_dim[:j + 1, update_pos]

    matrix_profile = np.sqrt(matrix_profile)
    if return_dimension:
        return matrix_profile, profile_index, profile_dimension
    else:
        return matrix_profile, profile_index


def rcmstamp(seq, sub_len, sea_ran, return_dimension=False):
    if sub_len < 4:
        raise RuntimeError('Subsequence length (sub_len) must be at least 4')
    # if sea_ran / sub_len < 2:
    #     raise RuntimeError('Search range (sea_ran) must be at least 2x subsequence length (sub_len))')

    exc_zone = sub_len // 2
    seq = np.array(seq, dtype=float, copy=True)

    if seq.ndim == 1:
        seq = np.expand_dims(seq, axis=0)

    seq_len = seq.shape[1]
    sub_num = seq.shape[1] - sub_len + 1  # n - l + 1
    n_dim = seq.shape[0]
    skip_loc = np.zeros(sub_num, dtype=bool)
    for i in range(sub_num):
        if not np.all(np.isfinite(seq[:, i:i + sub_len])):
            skip_loc[i] = True
    seq[~np.isfinite(seq)] = 0

    matrix_profile = np.empty((n_dim, sub_num))
    matrix_profile[:] = np.inf
    profile_index = -np.ones((n_dim, sub_num), dtype=int)
    #    seq_freq = np.empty((n_dim, sea_ran * 2), dtype=np.complex128)
    seq_mu = np.empty((n_dim, sub_num))
    seq_sig = np.empty((n_dim, sub_num))
    if return_dimension:
        profile_dimension = []
        for i in range(n_dim):
            profile_dimension.append(np.empty((i + 1, sub_num), dtype=int))
    for i in range(n_dim):
        seq_mu[i, :], seq_sig[i, :] = \
            _mass_pre(seq[i, :], sub_len)

    dist_profile = np.empty((n_dim, sub_num))
    que_sig = np.empty(n_dim)
    tic = time.time()
    for i in range(sub_num):
        cur_prog = (i + 1) / sub_num
        time_left = ((time.time() - tic) / (i + 1)) * (sub_num - i - 1)
        print('\rProgress [{0:<50s}] {1:5.1f}% {2:8.1f} sec'
              .format('#' * int(cur_prog * 50),
                      cur_prog * 100, time_left), end="")

        relative_start = max(0, i - sea_ran)
        relative_end = min(sub_num + 1, i + sea_ran - sub_len + 2)

        for j in range(n_dim):
            que = seq[j, i:i + sub_len]

            # Extract the sequence segment for -m, i, +m
            seq_seg = seq[j, max(0, i - sea_ran): min(seq_len, i + sea_ran + 1)]
            mu_seg = seq_mu[j, relative_start: relative_end]
            sig_seg = seq_sig[j, relative_start: relative_end]

            dist_profile[j, :] = np.inf
            dist_profile[j, relative_start: relative_end], que_sig = \
                _mass(seq_seg, que, sub_len, mu_seg, sig_seg)

        if skip_loc[i] or np.any(que_sig < _EPS):
            continue

        exc_zone_st = max(0, i - exc_zone)
        exc_zone_ed = min(sub_num, i + exc_zone)
        dist_profile[:, exc_zone_st:exc_zone_ed] = np.inf
        dist_profile[:, skip_loc] = np.inf
        dist_profile[seq_sig < _EPS] = np.inf

        dist_profile_dim = np.argsort(dist_profile, axis=0)
        dist_profile_sort = np.sort(dist_profile, axis=0)
        dist_profile_cumsum = np.zeros(sub_num)
        for j in range(n_dim):
            dist_profile_cumsum += dist_profile_sort[j, :]
            dist_profile_mean = dist_profile_cumsum / (j + 1)
            update_pos = dist_profile_mean < matrix_profile[j, :]
            profile_index[j, update_pos] = i
            matrix_profile[j, update_pos] = dist_profile_mean[update_pos]
            if return_dimension:
                profile_dimension[j][:, update_pos] = \
                    dist_profile_dim[:j + 1, update_pos]

    matrix_profile = np.sqrt(matrix_profile)
    if return_dimension:
        return matrix_profile, profile_index, profile_dimension
    else:
        return matrix_profile, profile_index