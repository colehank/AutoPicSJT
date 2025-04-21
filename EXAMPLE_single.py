# %%
from __future__ import annotations

import os

from wasabi import msg

import src
from src.pipeline import SituationProcessor
results_dir = 'results/final'
trait = 'N'
itemID = '1'
you = 'Ye'
trait_ = 'Neuroticism'

this_dir = f'{results_dir}/{trait}'
fig_dir = f'{this_dir}/figs'
data_dir = f'{this_dir}/data'
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

dm = src.DataManager()
situs = dm.read('situation_judgment_test', 'SJTs', extract_stiu=True)
situation = situs[trait][itemID]
# %%
P = SituationProcessor(model='gpt-4o',situ=situation, ref=you, trait=trait_)
res = P.fit(verbose=True)
#%%
fig_G = src.draw_G(P.G)
fig_Gs = src.draw_Gs(P.Gs)
fig_intergarted_Gs = src.draw_Gs(P.intergerated_Gs)
#%%
fig_G.savefig(f'{fig_dir}/G_{trait}_{itemID}.tif', dpi=300, bbox_inches='tight')
fig_Gs.savefig(f'{fig_dir}/Gs_{trait}_{itemID}.tif', dpi=300, bbox_inches='tight')
fig_intergarted_Gs.savefig(f'{fig_dir}/GsEnriched_{trait}_{itemID}.tif', dpi=300, bbox_inches='tight')
#%%
src.dic_G(P.intergerated_Gs['P'])
# %%
import pickle
with open('testP.pkl', 'wb') as f:
    pickle.dump(res, f)
