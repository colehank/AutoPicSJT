# %%
from __future__ import annotations

import numpy as np
from scipy.stats import pointbiserialr

def pomit_biserial(data):
    """
    Compute the discrimination index for each item using the point-biserial correlation.

    The discrimination index is measured by calculating the point-biserial correlation
    between the item scores (0 or 1) and the total score of the remaining items (i.e., the
    total score with the current item excluded). A higher correlation indicates that the
    item better distinguishes between respondents with higher and lower overall scores.

    Parameters
    ----------
    data : numpy.ndarray
        A 2D numpy array of shape (nsub, nitem), where each row corresponds to a subject and
        each column corresponds to an item. Each element should be scored as 0 or 1.

    Returns
    -------
    discrimination : numpy.ndarray
        A 1D numpy array of shape (nitem,) containing the discrimination index (point-biserial
        correlation coefficient) for each item.
    """
    nsub, nitem = data.shape
    discrimination = []

    for i in range(nitem):
        # Compute the total score for each subject, excluding the current item.
        total_excluded = data.sum(axis=1) - data[:, i]
        # Calculate the point-biserial correlation between the item's responses and the modified total score.
        r, p = pointbiserialr(data[:, i], total_excluded)
        discrimination.append([r,p])

    return discrimination

#%%
data = np.array([
    [1, 0, 1, 0],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 0, 1, 0],
])
pomit_biserial(data)
#%%
