# %%
import src
import os
import os.path as op
import pickle
import networkx as nx
from wasabi import msg
#%%
trait = 'N'
item_id = '2'

p = f'results/{trait}/{item_id}/{trait}_{item_id}.pkl'
with open(p, 'rb') as f:
    res = pickle.load(f)
situs = src.DataManager().read('situation_judgment_test', 'SJTs', extract_stiu=True)
diamonds = src.DataManager().read('situation_DIAMONDS', 'DIAMONDS')
#%%
from src import PromptTemplateManager
pm = PromptTemplateManager()
#%%
from src.prompts.diamonds import prompt_template
src.print_conversation(prompt_template)
#%%
from src.models.llms import TempletLLM
llm = TempletLLM('diamonds')
#%%
res = llm.call(situs['N']['1'], word = '@!ddf')
#%%
import numpy as np
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
#%%

