from __future__ import annotations

import numpy as np

from sta import reliability
from sta import utils
from sta import validity

# 创建示例数据
data = np.random.randn(100, 10)  # 100个被试，10个题目

# 计算信度
alpha = reliability.cronbach_alpha(data)
split_half_r, corrected_r = reliability.split_half_reliability(data)

# 计算效度
# 假设我们有一个效标分数
criterion_scores = np.random.randn(100)
validity_results = validity.criterion_validity(np.sum(data, axis=1), criterion_scores)

# 进行因子分析
factor_results = utils.factor_analysis(data)

# 计算描述性统计
stats = utils.descriptive_stats(data)
