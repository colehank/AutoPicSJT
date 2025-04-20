# %%
from __future__ import annotations

from wasabi import msg

import src
from src.pipeline import SituationProcessor

trait = 'N'
itemID = '0'
you = 'Ye'

dm = src.DataManager()
situs = dm.read('situation_judgment_test', 'SJTs', extract_stiu=True)
situation = situs[trait][itemID]
situation
# %%
P = SituationProcessor(model='chatgpt-4o',situ=situation, ref=you, trait='N')
res = P.fit(verbose=True)
# P.situ_graph()
# P.Gs_from_situ()
# P.extract_cues_from_Gs()
# P.enrich_Gs_by_cues()
# P.intergrate_enriched_Gs()
src.draw_Gs(P.Gs)
