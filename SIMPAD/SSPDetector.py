#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chun-Tung Li
"""

import copy
import numpy as np
import pandas as pd
from skimage import filters
from functools import reduce
from SIMPAD.RCMP import rcmstomp

def _generateGraph(var_len_valleys):
    vertexList = []
    for l in var_len_valleys:
        vertexList = vertexList + var_len_valleys[l]
    vertexList = np.array(vertexList)

    edgeList = []
    for i in np.arange(len(vertexList)):
        overlap = \
        np.where((~(vertexList[i, 0] > vertexList[:, 1]) & ~(vertexList[i, 1] < vertexList[:, 0])) == True)[0]
        overlap = np.delete(overlap, np.where(overlap == i)[0])
        for j in overlap:
            edgeList.append((i, j))
    edgeList = np.array(edgeList)

    adjacencyList = [[] for vartex in vertexList]
    for edge in edgeList:
        adjacencyList[edge[0]].append(edge[1])
    adjacencyList = np.array(adjacencyList)

    return vertexList, edgeList, adjacencyList


def _generateSubgraphs(vertextList, adjacencyList):
    subgraphs = []
    freeVertices = list(np.arange(len(vertextList)))
    while freeVertices:
        freeVertex = freeVertices.pop()
        subgraph = _constructSubgraph(freeVertex, adjacencyList, [freeVertex])
        freeVertices = [vertex for vertex in freeVertices if vertex not in subgraph]
        subgraphs.append(subgraph)
    return subgraphs


def _constructSubgraph(vertex, adjacencyList, subgraph):
    neighbors = [vertex for vertex in adjacencyList[vertex] if vertex not in subgraph]
    if (len(neighbors) == 0):
        return subgraph
    else:
        subgraph = subgraph + neighbors
        for vertex in neighbors:
            subgraph = _constructSubgraph(vertex, adjacencyList, subgraph)
        return subgraph


def _incumb(vertexWeight, adjacencyList):
    N = len(vertexWeight)

    X = np.zeros(N, dtype=bool)
    for i in range(N):
        if (len(adjacencyList[i]) == 0):
            X[i] = True

    Z = np.zeros(N)
    for i in range(N):
        Z[i] = vertexWeight[i] - np.sum(vertexWeight[list(adjacencyList[i])])

    freeVertices = np.where(X == 0)[0]
    while True:
        if len(freeVertices) == 0:
            break;
        imin = freeVertices[np.argmax(Z[freeVertices])]
        X[imin] = True
        freeVertices = freeVertices[freeVertices != imin]
        X[adjacencyList[imin]] = False
        freeVertices = freeVertices[~np.isin(freeVertices, adjacencyList[imin])]
        for i in freeVertices:
            Z[i] = vertexWeight[i] - np.sum(vertexWeight[np.intersect1d(freeVertices, adjacencyList[i])])
    return X

def _calculateLB(X, vertexWeight, adjacencyList, visitedVertices=[]):
    neighbors = np.array([], dtype=int)
    if (len(adjacencyList[np.where(X == 1)[0]]) > 0):
        neighbors = reduce(np.union1d, adjacencyList[np.where(X == 1)[0]])
    if (len(visitedVertices) > 0):
        neighbors = np.append(neighbors, visitedVertices[np.where(X[visitedVertices] == False)])
    neighbors = np.unique(neighbors)
    neighbors = np.array(neighbors, dtype=int)
    wj = np.sum(vertexWeight[neighbors])
    return -1 * (np.sum(vertexWeight) - wj)

def _BBND(vertexWeight, adjacencyList, LB, OPT_X):
    N = len(vertexWeight)
    X = np.zeros(N)
    X[:] = np.nan
    visitedVertices = np.array([], dtype=int)
    OPT = np.sum(vertexWeight[OPT_X == 1])
    prob = {'X': [], 'visitedVertices': []}
    sub_probs = []

    while True:
        if (np.sum(np.isnan(X)) == 0):
            if (np.sum(vertexWeight[np.where(X == 1)[0]]) > OPT):
                OPT = np.sum(vertexWeight[np.where(X == 1)[0]])
                OPT_X = X
            if (len(sub_probs) > 0):
                prob = sub_probs.pop()
                X = prob['X']
                visitedVertices = prob['visitedVertices']
            else:
                break

        for i in range(N):
            if (~np.any(X[list(adjacencyList[i])])):
                X[i] = 1
                if (not i in visitedVertices):
                    visitedVertices = np.append(visitedVertices, i)

        Z = np.zeros(N)
        for i in range(N):
            Z[i] = vertexWeight[i] - np.sum(vertexWeight[list(adjacencyList[i])])
        if (len(visitedVertices) > 0):
            Z[visitedVertices] = np.inf
        imin = np.argmin(Z)

        visitedVertices = np.append(visitedVertices, imin)

        X[imin] = 0
        LB0 = _calculateLB(X, vertexWeight, adjacencyList, visitedVertices)

        X[imin] = 1
        LB1 = _calculateLB(X, vertexWeight, adjacencyList, visitedVertices)

        if (LB0 < LB1):
            if (LB1 < LB):
                X[imin] = 1
                prob['X'] = X.copy()
                prob['visitedVertices'] = visitedVertices.copy()

                prob['X'][list(adjacencyList[imin])] = 0
                neighbors = adjacencyList[imin]
                for i in neighbors:
                    if (not i in prob['visitedVertices']):
                        prob['visitedVertices'] = np.append(prob['visitedVertices'], i)
                if (np.sum(np.isnan(prob['X'])) < 0):
                    sub_probs.append(prob.copy())

            X[imin] = 0
        else:
            if (LB0 < LB):
                X[imin] = 0
                prob['X'] = X.copy()
                prob['visitedVertices'] = visitedVertices.copy()
                if (np.sum(np.isnan(prob['X'])) < 0):
                    sub_probs.append(prob.copy())
            X[imin] = 1
            X[list(adjacencyList[imin])] = 0
            neighbors = adjacencyList[imin]
            for i in neighbors:
                if (not i in visitedVertices):
                    visitedVertices = np.append(visitedVertices, i)
    return OPT_X


def MWIS(vertexWeight, adjacencyList):
    '''
    :param vertexWeight: List of real-valued vertex weight
    :param adjacencyList: List of adjacency vertices
    :return: Maximum sum of weights of the independent set
    :Note:
        This is the implementation of the follow publication:

        Pardalos, P. M., & Desai, N. (1991). An algorithm for finding a maximum weighted independent set in an arbitrary graph.
        International Journal of Computer Mathematics, 38(3-4), 163-175.
    '''
    X = _incumb(vertexWeight, adjacencyList)
    LB = _calculateLB(X, vertexWeight, adjacencyList)
    return _BBND(vertexWeight, adjacencyList, LB, X)

def _findVallies(mp, l, sigma):
    vallies = []
    N = len(mp)
    isCounting = False
    startIdx = -1
    for i in range(N):
        if ((mp[i] <= sigma) and (i < N-1)):
            if not isCounting:
                startIdx = i
                isCounting = True
        else:
            if isCounting:
                isCounting = False
                endIdx = i
                if (endIdx - startIdx < l):
                    continue
                vallies.append([startIdx, endIdx, l, np.sum(sigma - mp[startIdx:endIdx])])
    return vallies


def SIMPAD(data, l, m, dimensions=None):
    '''
    :param data: d x n real-valued array - Input Time Series
    :param l: int - Target pattern length
    :param m: int - Maximum displacement between patterns
    :param dimensions: int (OPTIONAL) - The number of dimensions to be compared. It calculate the MP by all dimensions if not sepcified.
    :return: List of bool - Indicate SSP, True for identified SSP
    '''

    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    ndim = data.shape[0]

    mp, ip = rcmstomp(data, l, m)
    if dimensions:
        mp = pd.Series(mp[dimensions])
    else:
        mp = pd.Series(mp[mp.shape[0] - 1])

    mp[np.isinf(mp)] = np.nan
    mp = mp.interpolate()
    mp = mp.rolling(l, 1).mean()

    N = len(mp)

    sigma = filters.threshold_otsu(mp.dropna())
    valleys = _findVallies(mp, l, sigma)

    detection = np.zeros(N, dtype=bool)
    for valley in valleys:
        detection[int(valley[0]):int(valley[1])] = True

    return detection.tolist()


def mSIMPAD(data, L, m_factor, dimensions=None):
    '''
    :param data: d x n real-valued array - Input Time Series
    :param L: List of int - Target pattern lengths
    :param m_factor: int - A factor of target pattern length to determine the maximum displacement between patterns
    :param dimensions: int (OPTIONAL) - The number of dimensions to be compared. It calculate the MP by all dimensions if not sepcified.
    :return: List of bool - Indicate SSP, True for identified SSP
    '''

    var_len_valleys = {}

    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    ndim = data.shape[0]
    N = data.shape[1] - np.min(L) + 1

    for l in L:
        print('\nComputing RCMP of length ' + str(l))
        m = m_factor * l
        mp, ip = rcmstomp(data, l, m)
        if dimensions:
            mp = pd.Series(mp[dimensions])
        else:
            mp = pd.Series(mp[mp.shape[0] - 1])

        mp[np.isinf(mp)] = np.nan
        mp = mp.interpolate()
        mp = mp.rolling(l, 1).mean()
        mp = mp * np.sqrt(1 / l)

        sigma = filters.threshold_otsu(mp.dropna())
        var_len_valleys[l] = _findVallies(mp, l, sigma)

    vertexList, edgeList, adjacencyList = _generateGraph(var_len_valleys)
    subgraphs = _generateSubgraphs(vertexList, adjacencyList)

    solution = np.zeros(len(vertexList), dtype=bool)
    for subgraph in subgraphs:
        vl = np.array(copy.deepcopy(vertexList[subgraph, 3]))
        al = np.array(copy.deepcopy(adjacencyList[subgraph]))
        for i in range(len(al)):
            for j in range(len(al[i])):
                al[i][j] = np.where(subgraph == al[i][j])[0][0]
        OPT_X = MWIS(vl, al)
        solution[subgraph] = OPT_X

    valleys = vertexList[solution, :]
    detection = np.zeros(N, dtype=bool)
    for valley in valleys:
        detection[int(valley[0]):int(valley[1])] = True

    return detection.tolist()