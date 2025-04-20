# %%
from __future__ import annotations

import json
import os
import pickle
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor

from matplotlib import colormaps as cm
from tqdm.autonotebook import tqdm
from wasabi import msg

import src
from src import TempletLLM
# %%
BF_MAP = {
    'O': 'Openness',
    'C': 'Conscientiousness',
    'E': 'Extraversion',
    'A': 'Agreeableness',
    'N': 'Neuroticism',
}
def get_knowledge(G, situ = None, llm_correct = False):
    cue_types = ['att|obj', 'obj-obj', 'att|obj-obj', 'obj-att|obj', 'att|obj-att|obj']
    cues = {
        cue_type:src.extract_knowledge(
        G, cue_type=cue_type,
        ) for cue_type in cue_types
    }
    if llm_correct:
        cues = kg_corrector.call(situ=situ, cues=cues)
    return cues

def get_colors_by_cues(G, cues, cmap = 'gist_ncar'):

    def custom_cmap(value):
        colormap = cm.get_cmap(cmap)  # You can replace 'gist_ncar' with any colormap you prefer
        return colormap(value)

    colors_cue = [custom_cmap(i / len(cues)) for i in range(len(cues))]
    gh_id = [src.map_knowledge(G, cue['content'], cue['type']) for cue in cues]
    colors = {'edge': {}, 'label': {}}
    for i, cue in enumerate(gh_id):
        color = colors_cue[i]
        for j in cue:
            if isinstance(j, tuple): # 处理关系边
                colors['edge'][j] = color
                colors['label'][j] = color
            if 'attribute' in j: # 处理属性边
                colors['edge'][(j, f'object_{j.split("|")[-2]}')] = color
                colors['label'][(j, f'object_{j.split("|")[-2]}')] = color
    return colors

def draw_G_cue_highlight(G, cues, cmap = 'hsv', title = 'SituCues'):
    colors = get_colors_by_cues(G, cues, cmap=cmap)
    fig = src.draw_G(
        G,
        colors=colors,
        attribute_edge_color='black',
        attribute_node_shape='^',
        object_node_color='lightgray',
        attribute_node_color='lightgray',
        title = title,
    )
    return fig

#%%
if __name__ == '__main__':

    results_dir = 'results/trait_cues'
    fig_dir_all = f'{results_dir}/figs/all'
    fig_dir_clean = f'{results_dir}/figs/clean'
    os.makedirs(fig_dir_all, exist_ok=True)
    os.makedirs(fig_dir_clean, exist_ok=True)
    traits = ['O', 'C', 'E', 'A', 'N']
    n_item = 21

    dm = src.DataManager()
    kg_corrector = TempletLLM('kg_correct')
    cues_extractor = TempletLLM('trait_extraction')
    cues_extractor.llm.model = 'gpt-4o'

    situs = dm.read('situation_judgment_test', 'SJTs', extract_stiu=True, replace_you=True)
    G = dm.read('situation_judgement_test_G', 'G')
    Gs = dm.read('situation_judgement_test_G', 'Gs')

    situs = {
        trait: {int(k): v for k, v in items.items()}
        for trait, items in situs.items()
    }
    graphs = {
        trait:{
            item_id:get_knowledge(G[f'{trait}_{item_id}'])
            for item_id in range(n_item+1)
        } for trait in traits
    }
    # %%
    # API calling
    def retry_forever(func):
        def wrapper(*args, **kwargs):
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f'[RETRY] Error: {e}')
        return wrapper

    @retry_forever
    def task(trait, item_id):
        graph = graphs[trait][item_id]
        passage = situs[trait][item_id]
        passage.replace(
        'your', "Ye's",
        ).replace(
            'you', 'Ye',
        ).replace(
                'are', 'is',
        ).replace(
                    'You', 'Ye',
        ).replace(
                        'Your', "Ye's",
        )
        while True:
            try:
                cues = cues_extractor.call(passage=passage, trait=BF_MAP[trait], graph=graph)['cues']
                if len(cues) == 0 or cues == None:
                    msg.warning(
                        f'Trait cues returned None for {trait}_{item_id}, retrying...',
                    )
                    continue
                break
            except Exception as e:
                print(f'[RETRY] Error: {e}')
                continue
        return trait, item_id, cues

    # 构建任务列表
    tasks = [(trait, item_id) for trait in traits for item_id in range(n_item + 1)]
    all_cues = {trait: {} for trait in traits}

    # 使用线程池 + 进度条并发执行任务
    with ThreadPoolExecutor(max_workers=110) as executor:
        futures = {executor.submit(task, trait, item_id): (trait, item_id) for trait, item_id in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Extracting cues'):
            trait, item_id, cues = future.result()
            all_cues[trait][item_id] = cues

    # %%
    with open(f'{results_dir}/trait_cues.json') as f:
        all_cues = json.load(f)
    clean_Gs = {
        f'{trait}_{item_id}': src.utils.entry_build_graph(cues)
        for trait in traits
        for item_id, cues in all_cues[trait].items()
    }
    # plotting
    for trait in traits:
        os.makedirs(f'{fig_dir_all}/{trait}', exist_ok=True)
        os.makedirs(f'{fig_dir_clean}/{trait}', exist_ok=True)
        for item_id in range(n_item + 1):
            note = f'*Situation Cues - {trait}_{item_id} '
            print(f'\r{note}', flush=True, end='')
            this_G = G[f'{trait}_{item_id}']
            cues = all_cues[trait][str(item_id)]
            fig = draw_G_cue_highlight(this_G, cues, cmap='hsv', title=None)
            fig.savefig(
                f'{results_dir}/figs/all/{trait}/{trait}_{item_id}.tif', dpi=300,
                bbox_inches='tight',
                # transparent = True,
            )
            clean_G = clean_Gs[f'{trait}_{item_id}']
            fig_clean = src.draw_G(clean_G, title=note, dpi=300)
            fig_clean.savefig(f'{results_dir}/figs/clean/{trait}/{trait}_{item_id}.tif', dpi=300, bbox_inches='tight')
#%%
    with open(f'{results_dir}/trait_cues.json', 'w') as f:
        json.dump(all_cues, f, indent=4)
    with open(f'{results_dir}/trait_cues_clean.pkl', 'wb') as f:
        pickle.dump(clean_Gs, f)
#%%
    # written cues to G
