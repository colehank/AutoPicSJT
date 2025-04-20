"""
信度计算方法模块
包含分半信度、克朗巴赫系数、重测信度等计算方法
"""
from __future__ import annotations

from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from scipy import stats

def split_half_reliability(
    data: np.ndarray | pd.DataFrame,
    method: str = 'odd-even',
) -> tuple[float, float]:
    """
    计算分半信度

    参数:
        data: 数据矩阵，每行是一个被试，每列是一个题目
        method: 分半方法，可选 'odd-even'（奇偶分半）或 'random'（随机分半）

    返回:
        (分半信度系数, 校正后的信度系数)
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    n_items = data.shape[1]

    if method == 'odd-even':
        # 奇偶分半
        odd_items = data[:, ::2]
        even_items = data[:, 1::2]
    else:
        # 随机分半
        indices = np.random.permutation(n_items)
        mid = n_items // 2
        odd_items = data[:, indices[:mid]]
        even_items = data[:, indices[mid:]]

    # 计算两半的得分
    odd_scores = np.sum(odd_items, axis=1)
    even_scores = np.sum(even_items, axis=1)

    # 计算相关系数
    r = stats.pearsonr(odd_scores, even_scores)[0]

    # 使用Spearman-Brown公式校正
    corrected_r = (2 * r) / (1 + r)

    return r, corrected_r

def cronbach_alpha(data: np.ndarray | pd.DataFrame) -> float:
    """
    计算克朗巴赫系数（Cronbach's α）

    参数:
        data: 数据矩阵，每行是一个被试，每列是一个题目

    返回:
        克朗巴赫系数
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    n_items = data.shape[1]
    item_var = np.var(data, axis=0)
    total_var = np.var(np.sum(data, axis=1))

    alpha = (n_items / (n_items - 1)) * (1 - np.sum(item_var) / total_var)
    return alpha

def test_retest_reliability(
    data1: np.ndarray | pd.DataFrame,
    data2: np.ndarray | pd.DataFrame,
) -> float:
    """
    计算重测信度

    参数:
        data1: 第一次测试的数据
        data2: 第二次测试的数据

    返回:
        重测信度系数
    """
    if isinstance(data1, pd.DataFrame):
        data1 = data1.values
    if isinstance(data2, pd.DataFrame):
        data2 = data2.values

    # 计算总分
    scores1 = np.sum(data1, axis=1)
    scores2 = np.sum(data2, axis=1)

    # 计算相关系数
    r = stats.pearsonr(scores1, scores2)[0]
    return r
