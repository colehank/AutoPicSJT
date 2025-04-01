# %%
import src
from src.models import TempletLLM
from src.datasets import DataManager
import json
from tqdm.autonotebook import tqdm
#%%
items = DataManager().read('situation_judgment_test', 'SJTs', extract_stiu=True)

vng_llm = TempletLLM('vng_generation')
ner_llm = TempletLLM('NER')
tri_llm = TempletLLM('triple_extraction')
kg_llm = TempletLLM('kg_generation')
#%%
vng_res = vng_llm.call(items['N']['4'])
ner_res = ner_llm.call(items['N']['4'])
tri_res = tri_llm.call(items['N']['4'], named_entities = ner_res)
kg_res = kg_llm.call(items['N']['4'], triples = tri_res)
#%%
from src.viz import kg
fig = kg.draw_kg(
    tri_res['triples'],
    node_fontsize=10,
    node_size=500,
    edge_fontsize=8,
)
#%%
def ner_per_vng(vng_res):
    vng = vng_res['VNG']
    ner_res = {
        key: ner_llm.call(vng[key])
        for key in tqdm(vng.keys(), decs='Extracting named entities')
    }
    return ner_res

def tri_per_vng(vng_res, ner_res):
    vng = vng_res['VNG']
    tri_res = {
        key: tri_llm.call(vng[key], named_entities=ner_res[key])
        for key in tqdm(vng.keys(), desc='Extracting triples')
    }
    return tri_res

ner_vng = ner_per_vng(vng_res)
tri_vng = tri_per_vng(vng_res, ner_vng)
#%%
kg.draw_kg(
    tri_vng['P']['triples'],
    node_fontsize=10,
    node_size=500,
    edge_fontsize=8,
)