# %%
from __future__ import annotations

import json
import pickle

import networkx as nx

import src
from src import TempletLLM
# %%
emo_llm = TempletLLM('emotion_analysis')
exp_llm = TempletLLM('emotion_to_expression')
cls_llm = TempletLLM('classfy_node')

dm = src.DataManager()

situs, G, clean_G = (
    dm.read('situation_judgment_test', 'SJTs', extract_stiu=True, replace_you=True),
    dm.read('situation_judgement_test_G', 'G'),
    dm.read('situation_judgement_test_G', 'clean_G'),
)

img_schema = {
    'base': dm.read('image_schema', 'image_schema'),
    'emotion': dm.read('image_schema', 'emotion_experssion'),
    'scene': dm.read('image_schema', 'scene_schema'),
    'object': dm.read('image_schema', 'object_schema'),
}

trait_cues = json.load(open('results/trait_cues/trait_cues.json'))
vng_Gs = pickle.load(open('results/vng_graph/vng_Gs.pkl', 'rb'))
# %%
trait, item_id = 'N', '1'
idx = f'{trait}_{item_id}'

situ = situs[trait][item_id]
nodes_info = dict(G[idx].nodes(data=True))
cls_node = cls_llm.call(passage=situ, nodes=nodes_info)
cls_node
# %%
cues = trait_cues[trait][item_id]
vng_G = vng_Gs[idx]

cue_VNG = src.which_vng_for_cues(vng_G, cues)
todo_VNG = [vng_G[i] for i in cue_VNG]
# %%
def make_expression(situation, trait, ana_character, act_character):
    # 1. Emotion Analysis
    emotion = emo_llm.call(
        passage=situation, trait=trait,
        analyze_character=ana_character,
        activate_character=act_character,
    )['emotion']['emotion']
    # 2. Emotion to Expression
    expression = exp_llm.call(passage=situation, emotion=emotion, character=ana_character)
    return expression

ye_exp = make_expression(situ, trait, 'Ye', 'Ye')
fr_exp = make_expression(situ, trait, 'friend', 'Ye')
woman_exp = make_expression(situ, trait, 'woman', 'Ye')
graph_language = TempletLLM('graph_to_language')
knowledge = src.get_knowledge(vng_Gs[idx]['E'])
description = graph_language.call(passage=knowledge)
#%%
# prompt = f'{character} {scene} {character_expression} {scene_expression}'
