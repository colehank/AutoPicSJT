# %%
from __future__ import annotations

from wasabi import msg

import src
from src.pipeline import SituationProcessor

trait = 'N'
itemID = '0'
you = 'Ye'
dm = src.DataManager()
situs = dm.read('situation_judgment_test', 'SJTs', extract_stiu=True, replace_you=True)
situation = situs[trait][itemID]
situation
# %%
processor = SituationProcessor(model='gpt-4o',situ=situation, ref=you, trait='Neuroticism', debug=True)
res = processor.fit()
# res = processor.fit(situation, trait)
# G = processor.situ_graph(situation)
# cues = processor.cues_extraction(situation, G, trait)
# cues_enrich = processor.cues_enrich(situation, cues, trait)

#%%
res = processor.fit(situation, trait)

#%%
from src import which_vng_for_cues
cue_idx = which_vng_for_cues(res['vng_graphs'], res['cues'])
cue_Gs = [res['vng_graphs'][idx] for idx in cue_idx]
src.draw_G(cue_Gs[1])
# %%
VNG_Gs = res['vng_graphs']
cues = res['cues']
enriched_cues = res['enriched_cues']

matchs = which_vng_for_cues(VNG_Gs, cues)
matched_Gs = {idx:VNG_Gs[idx] for idx in matchs}
ori_G = res['situation_graph']
src.draw_G(ori_G)
src.draw_Gs(matched_Gs)
# %%
from src import find_node_by_value

def get_max_attribute(G, object_id):
    object_num = int(object_id.split('_')[1])
    max_attr_idx = 0
    for node in G.nodes():
        if isinstance(node, str) and node.startswith('attribute|'):
            parts = node.split('|')
            if len(parts) == 3:
                node_obj_idx = int(parts[1])
                attr_idx = int(parts[2])
                if node_obj_idx == object_num and attr_idx > max_attr_idx:
                    max_attr_idx = attr_idx

    return max_attr_idx


for idx, G in matched_Gs.items():
    new_G = G.copy()
    for e_node_type, content in enriched_cues.items():
        if e_node_type == 'character':
            for cha, exp in content.items():
                print(f'--Character-{cha}--')

                G_node_id = find_node_by_value(new_G, cha, 'object_node')
                G_node_num = int(G_node_id.split('_')[1])

                print(f'G_node_id: {G_node_id}, G_node_num: {G_node_num}')
                max_attr_idx = get_max_attribute(ori_G, G_node_id) # VNG的G的attribute id继承自原始的G
                print(f'max_attr_idx: {max_attr_idx}')
                body_add_attr_id = f'attribute|{G_node_num}|{get_max_attribute(new_G, G_node_id) + 1}'
                face_add_attr_id = f'attribute|{G_node_num}|{get_max_attribute(new_G, G_node_id) + 2}'

                new_G.add_node(body_add_attr_id, value='_exp_body', type='attribute_node', content = exp['body'])
                new_G.add_edge(body_add_attr_id, G_node_id, type='attribute_edge')

                new_G.add_node(face_add_attr_id, value='_exp_face', type='attribute_node', content = exp['facial'])
                new_G.add_edge(face_add_attr_id, G_node_id, type='attribute_edge')

        elif e_node_type == 'scene':
            for scene, exp in content.items():
                print(f'--Scene-{scene}--')

                G_node_id = find_node_by_value(new_G, scene, 'object_node')
                G_node_num = int(G_node_id.split('_')[1])

                print(f'G_node_id: {G_node_id}, G_node_num: {G_node_num}')
                max_attr_idx = get_max_attribute(ori_G, G_node_id)
                print(f'max_attr_idx: {max_attr_idx}')
