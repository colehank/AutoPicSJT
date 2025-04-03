# %%
from __future__ import annotations

import os
import os.path as op
import pickle

import networkx as nx
import numpy as np
from wasabi import msg

import src
from src import PromptTemplateManager
from src.models.llms import TempletLLM
from src.prompts.diamonds import prompt_template
# %%
trait = 'N'
item_id = '2'

p = f'results/{trait}/{item_id}/{trait}_{item_id}.pkl'
with open(p, 'rb') as f:
    res = pickle.load(f)
situs = src.DataManager().read('situation_judgment_test', 'SJTs', extract_stiu=True)
diamonds = src.DataManager().read('situation_DIAMONDS', 'DIAMONDS')
# %%
pm = PromptTemplateManager()
# %%
src.print_conversation(prompt_template)
# %%
llm = TempletLLM('diamonds')
# %%
res = llm.call(situs['N']['1'], word='@!ddf')
# %%


def calculate_dimension_means(data):
    all_dim = np.unique(np.array([i.split('_')[0] for i in data.keys()]))
    means = []
    for dim in ['D', 'I', 'A', 'M', 'O', 'N', 'Dc', 'S']:
        dim_values = []
        for k, v in data.items():
            if k.startswith(dim):
                dim_values.append(v)
        means.append(np.mean(dim_values))
    return np.array(means)


means = calculate_dimension_means(res['DIAMONDS'])
print(means)
# %%
