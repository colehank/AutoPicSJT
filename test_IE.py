# %%
from src.models import TempletLLM
from src.datasets import DataManager
import src
import json
#%%
items = DataManager().read('situation_judgment_test', 'SJTs', extract_stiu=True)
t_itms = items['N']['4'].replace("your", "Ye's").replace("you", "Ye")
#%%
sg_llm = TempletLLM('sg_generation')
sg_llm.llm.model = 'claude-3-5-sonnet-latest'
#%%
sg_llm.llm
res = sg_llm.call(t_itms, json=True)
#%%
G = src.build_G(res['SceneGraph'])
src.print_G(G)
fig = src.draw_G(G,title='N_4')
fig
#%%
vng_llm = TempletLLM('vng_generation')
vng_llm.llm.model = 'claude-3-5-sonnet-latest'
res = vng_llm.call(t_itms)