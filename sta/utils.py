"""
工具函数模块
包含相关性分析、因子分析、描述性统计等通用函数
"""
from __future__ import annotations

from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from scipy import stats
from sklearn.preprocessing import StandardScaler

def correlation_matrix(
    data: np.ndarray | pd.DataFrame,
    method: str = 'pearson',
) -> tuple[np.ndarray, np.ndarray]:
    """
    计算相关系数矩阵

    参数:
        data: 数据矩阵，每行是一个被试，每列是一个变量
        method: 相关系数计算方法，可选 'pearson'、'spearman' 或 'kendall'

    返回:
        (相关系数矩阵, p值矩阵)
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    n_vars = data.shape[1]
    corr_matrix = np.zeros((n_vars, n_vars))
    p_matrix = np.zeros((n_vars, n_vars))

    for i in range(n_vars):
        for j in range(i, n_vars):
            if method == 'pearson':
                corr, p = stats.pearsonr(data[:, i], data[:, j])
            elif method == 'spearman':
                corr, p = stats.spearmanr(data[:, i], data[:, j])
            else:  # kendall
                corr, p = stats.kendalltau(data[:, i], data[:, j])

            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
            p_matrix[i, j] = p
            p_matrix[j, i] = p

    return corr_matrix, p_matrix

def factor_analysis(
    data: np.ndarray | pd.DataFrame,
    n_factors: int = None,
    rotation: str = 'varimax',
) -> dict:
    """
    进行因子分析

    参数:
        data: 数据矩阵，每行是一个被试，每列是一个变量
        n_factors: 因子数量，如果为None则自动确定
        rotation: 旋转方法，可选 'varimax'、'promax' 或 None

    返回:
        包含因子分析结果的字典
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    # 标准化数据
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 确定因子数
    if n_factors is None:
        fa = FactorAnalyzer(rotation=None, n_factors=data.shape[1])
        fa.fit(data_scaled)
        eigenvalues = fa.eigenvalues_
        n_factors = sum(eigenvalues > 1)

    # 进行因子分析
    fa = FactorAnalyzer(rotation=rotation, n_factors=n_factors)
    fa.fit(data_scaled)

    # 获取结果
    loadings = fa.loadings_
    variance = fa.get_factor_variance()
    kmo_all, kmo_model = fa.kmo()
    chi_square_value, p_value = fa.bartlett()

    return {
        'loadings': loadings,
        'variance': variance,
        'kmo': kmo_model,
        'bartlett_chi_square': chi_square_value,
        'bartlett_p_value': p_value,
        'n_factors': n_factors,
    }

def descriptive_stats(data: np.ndarray | pd.DataFrame) -> dict:
    """
    计算描述性统计量

    参数:
        data: 数据矩阵，每行是一个被试，每列是一个变量

    返回:
        包含描述性统计量的字典
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    stats_dict = {}
    for i in range(data.shape[1]):
        var_data = data[:, i]
        stats_dict[f'variable_{i+1}'] = {
            'mean': np.mean(var_data),
            'median': np.median(var_data),
            'std': np.std(var_data, ddof=1),
            'skewness': stats.skew(var_data),
            'kurtosis': stats.kurtosis(var_data),
            'min': np.min(var_data),
            'max': np.max(var_data),
            'q1': np.percentile(var_data, 25),
            'q3': np.percentile(var_data, 75),
        }

    return stats_dict
