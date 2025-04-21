# %%
from __future__ import annotations

import json

import networkx as nx

import src
from src import TempletLLM
# %%
def parse_by_type(type_str, data):
    segments = type_str.split('-')

    flat_slots = []

    for i, seg in enumerate(segments):
        flat_slots += seg.split('|')
        if i < len(segments) - 1:
            flat_slots.append('rel')

    result = {'obj': [], 'rel': [], 'att': []}

    for slot, value in zip(flat_slots, data):
        if slot == 'obj':
            result['obj'].append(value)
        elif slot == 'att':
            result['att'].append(value)
        elif slot == 'rel':
            result['rel'].append(value)

    return result

def label_cues(cues, G):
    new_G = G.copy()

    def add_activator(item, act):
        current = item.get('activator')
        if current is None:
            item['activator'] = [act]
        elif isinstance(current, list):
            current.append(act)
        else:
            item['activator'] = [current, act]

    for i, cue in enumerate(cues):
        act = i
        cue_id = src.map_knowledge(new_G, cue['content'], cue['type'])
        parsed = parse_by_type(cue['type'], cue_id)
        for part, vals in parsed.items():
            for val in vals:
                if part in ('obj', 'att'):
                    add_activator(new_G.nodes[val], act)
                elif part == 'rel':
                    add_activator(new_G.edges[val], act)
    return new_G

# %%
dm = src.DataManager()
kg_corrector = TempletLLM('kg_correct')

situs = dm.read('situation_judgment_test', 'SJTs', extract_stiu=True, replace_you=True)
G = dm.read('situation_judgement_test_G', 'G')
clean_G = dm.read('situation_judgement_test_G', 'clean_G')

with open('results/trait_cues/trait_cues.json') as f:
    trait_cues = json.load(f)
#%%
G_cued = {}
for trait, items in trait_cues.items():
    G_cued[trait] = {}
    for item_id, cues in items.items():
        this_G = G[f'{trait}_{item_id}']
        new_G = label_cues(cues, this_G)
        G_cued[trait][item_id] = new_G
