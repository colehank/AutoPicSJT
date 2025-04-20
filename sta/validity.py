"""
效度计算方法模块
包含效标效度、收敛效度、区分效度等计算方法
"""
from __future__ import annotations

from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from scipy import stats
from sklearn.preprocessing import StandardScaler

def criterion_validity(
    test_scores: np.ndarray | pd.Series,
    criterion_scores: np.ndarray | pd.Series,
) -> dict[str, float]:
    """
    计算效标效度

    参数:
        test_scores: 测验得分
        criterion_scores: 效标得分

    返回:
        包含同时效度和预测效度的字典
    """
    if isinstance(test_scores, pd.Series):
        test_scores = test_scores.values
    if isinstance(criterion_scores, pd.Series):
        criterion_scores = criterion_scores.values

    # 计算同时效度（Pearson相关系数）
    concurrent_validity = stats.pearsonr(test_scores, criterion_scores)[0]

    # 计算预测效度（如果数据是时间序列）
    if len(test_scores) > 1:
        # 使用滞后一期的相关系数作为预测效度
        predictive_validity = stats.pearsonr(test_scores[:-1], criterion_scores[1:])[0]
    else:
        predictive_validity = None

    return {
        'concurrent_validity': concurrent_validity,
        'predictive_validity': predictive_validity,
    }

def convergent_validity(
    data: np.ndarray | pd.DataFrame,
    factors: list[list[int]],
) -> dict[str, float]:
    """
    计算收敛效度

    参数:
        data: 数据矩阵，每行是一个被试，每列是一个题目
        factors: 因子结构，每个因子包含的题目索引列表

    返回:
        包含各因子收敛效度的字典
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    # 标准化数据
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 进行因子分析
    fa = FactorAnalyzer(rotation=None, n_factors=len(factors))
    fa.fit(data_scaled)

    # 计算因子载荷
    loadings = fa.loadings_

    # 计算每个因子的平均变异抽取量（AVE）
    convergence = {}
    for i, factor in enumerate(factors):
        factor_loadings = loadings[factor, i]
        ave = np.mean(factor_loadings ** 2)
        convergence[f'factor_{i+1}'] = ave

    return convergence

def discriminant_validity(
    data: np.ndarray | pd.DataFrame,
    factors: list[list[int]],
) -> dict[str, float]:
    """
    计算区分效度

    参数:
        data: 数据矩阵，每行是一个被试，每列是一个题目
        factors: 因子结构，每个因子包含的题目索引列表

    返回:
        包含各因子间区分效度的字典
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    # 标准化数据
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 进行因子分析
    fa = FactorAnalyzer(rotation=None, n_factors=len(factors))
    fa.fit(data_scaled)

    # 计算因子载荷
    loadings = fa.loadings_

    # 计算因子间相关系数
    factor_scores = fa.transform(data_scaled)
    factor_corr = np.corrcoef(factor_scores.T)

    # 计算每个因子的平均变异抽取量（AVE）
    discriminant = {}
    for i, factor1 in enumerate(factors):
        for j, factor2 in enumerate(factors):
            if i < j:  # 只计算上三角矩阵
                factor1_loadings = loadings[factor1, i]
                factor2_loadings = loadings[factor2, j]
                ave1 = np.mean(factor1_loadings ** 2)
                ave2 = np.mean(factor2_loadings ** 2)
                corr = factor_corr[i, j]

                # 计算区分效度（AVE的几何平均数与因子相关系数的比较）
                discriminant[f'factor_{i+1}_{j+1}'] = {
                    'correlation': corr,
                    'sqrt_ave': np.sqrt(ave1 * ave2),
                    'discriminant_valid': np.sqrt(ave1 * ave2) > abs(corr),
                }

    return discriminant

def construct_validity(
    data: np.ndarray | pd.DataFrame,
    n_factors: int = None,
) -> dict[str, float]:
    """
    计算构念效度

    参数:
        data: 数据矩阵，每行是一个被试，每列是一个题目
        n_factors: 因子数量，如果为None则自动确定

    返回:
        包含构念效度指标的字典
    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    # 标准化数据
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 进行因子分析
    if n_factors is None:
        # 使用Kaiser准则确定因子数
        fa = FactorAnalyzer(rotation=None, n_factors=data.shape[1])
        fa.fit(data_scaled)
        eigenvalues = fa.eigenvalues_
        n_factors = sum(eigenvalues > 1)

    fa = FactorAnalyzer(rotation='varimax', n_factors=n_factors)
    fa.fit(data_scaled)

    # 计算KMO值
    kmo_all, kmo_model = fa.kmo()

    # 计算Bartlett球形检验
    chi_square_value, p_value = fa.bartlett()

    return {
        'kmo': kmo_model,
        'bartlett_chi_square': chi_square_value,
        'bartlett_p_value': p_value,
        'n_factors': n_factors,
    }
