# %%
from __future__ import annotations

import copy
import json
import os
import os.path as op
import pickle
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from wasabi import msg

import src
from src.datasets import DataManager
from src.models import TempletLLM
plt.rcParams['font.family'] = 'Comic Sans MS'
plt.rcParams['font.family'] = 'Times New Roman'
# %%
sg_llm = TempletLLM('sg_generation')
sg_llm.llm.model = 'claude-3-5-sonnet-latest'
vng_llm = TempletLLM('vng_generation')
vng_llm.llm.model = 'claude-3-5-sonnet-latest'
# %%
def pipeline(trait, item_id, model='claude-3-5-sonnet-latest'):
    itemID = f'{trait}_{item_id}'
    items = DataManager().read('situation_judgment_test', 'SJTs', extract_stiu=True)
    t_itm = items[trait][item_id].replace(
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
    note = f"*PersonalitySJT's SceneGraph - {itemID}"

    sg_llm = TempletLLM('sg_generation')
    sg_llm.llm.model = model
    vng_llm = TempletLLM('vng_generation')
    vng_llm.llm.model = model

    def try_call(llm, name, situ, max_attempts=10000):
        for attempt in range(max_attempts):
            result = llm.call(situ)
            if result is not None:
                return result
            msg.warn(
                f'{name} generation attempt {attempt+1}/{max_attempts} failed for {itemID}, retrying...',
            )
        msg.fail(
            f'{name} generation failed after {max_attempts} attempts for {itemID}, skipping...',
        )
        raise RuntimeError(
            f'{name} generation failed after {max_attempts} attempts for {itemID}',
        )
        return None

    res_sg = try_call(sg_llm, 'SceneGraph', t_itm)
    res_vng = try_call(vng_llm, 'VNG', t_itm)
    res_vng_sg = {}
    for vng, content in res_vng['VNG'].items():
        this_sg = try_call(sg_llm, 'SceneGraph', content)
        res_vng_sg[vng] = (this_sg)

    G = src.build_G(res_sg['SceneGraph'])
    Gs = {
        vng: src.build_G(content['SceneGraph'])
        for vng, content in res_vng_sg.items()
    }

    fig_G = src.draw_G(G, title=note, dpi=300)
    fig_Gs = src.viz.draw_Gs(Gs)
    fig_Gs.text(0.5, 0,  note, fontsize=10, ha='center', va='bottom')
    plt.close('all')
    return {
        'itemID': itemID,
        'G': G,
        'Gs': Gs,
        'res_sg': res_sg,
        'res_vng': res_vng,
        'res_vng_sg': res_vng_sg,
        'fig_G': fig_G,
        'fig_Gs': fig_Gs,
    }


def save_results(res, result_dir):
    trait = res['itemID'].split('_')[0]
    item_id = res['itemID'].split('_')[1]
    os.makedirs(op.join(result_dir, trait), exist_ok=True)
    os.makedirs(op.join(result_dir, trait, item_id), exist_ok=True)
    res['fig_G'].savefig(
        op.join(
            result_dir, trait, item_id,
            f"SceneGraph_{res['itemID']}.png",
        ), dpi=300,
    )
    res['fig_Gs'].savefig(
        op.join(
            result_dir, trait, item_id,
            f"SceneGraph_VNG_{res['itemID']}.png",
        ), dpi=300,
    )
    save_res = copy.deepcopy(res)
    del save_res['fig_G']
    del save_res['fig_Gs']
    pickle_path = op.join(result_dir, trait, item_id, f"{res['itemID']}.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(save_res, f)


# %%
if __name__ == '__main__':
    # for trait in tqdm(['O', 'C', 'E', 'A', 'N'], desc='Running traits', position=0):
    #     for item_id in tqdm(range(0, 22), desc=f'Running items for {trait}', position=1):
    #         # Check if result already exists
    #         result_path = op.join(
    #             'results', trait, str(
    #                 item_id,
    #             ), f'{trait}_{item_id}.pkl',
    #         )
    #         if op.exists(result_path):
    #             continue
    #         # Try up to 100 times
    #         max_attempts = 10000
    #         for attempt in range(max_attempts):
    #             try:
    #                 res = pipeline(trait, str(item_id))
    #                 save_results(res, 'results')
    #                 break
    #             except Exception as e:
    #                 if attempt < max_attempts - 1:
    #                     msg.warn(
    #                         f'Attempt {attempt+1}/{max_attempts} for {trait}_{item_id} failed: {e}. Retrying...',
    #                     )
    #                 else:
    #                     msg.fail(
    #                         f'Failed to process {trait}_{item_id} after {max_attempts} attempts: {e}',
    #                     )
    # msg.good('Ha! All tasks completed successfully!')
    max_attempts = 10000
    num_workers = 110
    def run_task(trait, item_id):
        result_path = op.join('results', trait, str(item_id), f'{trait}_{item_id}.pkl')
        if op.exists(result_path):
            return f'{trait}_{item_id} skipped (already exists)'

        for attempt in range(max_attempts):
            try:
                res = pipeline(trait, str(item_id))
                save_results(res, 'results')
                return f'{trait}_{item_id} completed'
            except Exception as e:
                if attempt < max_attempts - 1:
                    msg.warn(f'Attempt {attempt+1}/{max_attempts} for {trait}_{item_id} failed: {e}. Retrying...')
                else:
                    msg.fail(f'Failed to process {trait}_{item_id} after {max_attempts} attempts: {e}')
                    return f'{trait}_{item_id} failed'

    # 构造所有任务
    tasks = [(trait, item_id) for trait in ['O', 'C', 'E', 'A', 'N'] for item_id in range(22)]
    # %%
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(run_task, trait, item_id): (trait, item_id) for trait, item_id in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing all items'):
            result_msg = future.result()
            print(result_msg)

    msg.good('Ha! All tasks completed successfully!')


# %%
    for trait in ['O', 'C', 'E', 'A', 'N']:
        for item_id in range(0, 22):
            note = f"*PersonalitySJT's EventGraph - {trait}_{item_id} "
            print(f'\r{note}', flush=True, end='')
            result_path = op.join(
                'results/event_graph', trait,
                str(item_id), f'{trait}_{item_id}.pkl',
            )
            with open(result_path, 'rb') as f:
                res = pickle.load(f)
            G = res['G']
            Gs = res['Gs']
            fig_G = src.draw_G(
                G,
                   title=None,
                #    title = note,
                   dpi=300,
                   attribute_edge_color='black',
                    attribute_node_shape='^',
                    object_node_color='lightgray',
                    attribute_node_color='lightgray',
            )
            fig_Gs = src.viz.draw_Gs(Gs)
            fig_Gs.text(0.5, 0, note, fontsize=10, ha='center', va='bottom')
            fig_G.savefig(
                op.join(
                    'results/event_graph', trait, str(item_id),
                    f"EventGraph_{res['itemID']}.tif",
                ), dpi=300, bbox_inches='tight',
            )
            fig_Gs.savefig(
                op.join(
                    'results/event_graph', trait, str(item_id),
                    f"EventGraph_VNG_{res['itemID']}.tif",
                ), dpi=300, bbox_inches='tight',
            )
            plt.close('all')
# %%
    Gs = {}
    G = {}
    for trait in ['O', 'C', 'E', 'A', 'N']:
        for item_id in range(0, 22):
            result_path = op.join(
                'results/event_graph', trait,
                str(item_id), f'{trait}_{item_id}.pkl',
            )
            with open(result_path, 'rb') as f:
                res = pickle.load(f)
            G[res['itemID']] = res['G']
            Gs[res['itemID']] = res['Gs']


    with open(
        op.join('src/datasets', 'G.pkl'), 'wb',
    ) as f:
        pickle.dump(G, f)
    with open(
        op.join('src/datasets', 'Gs.pkl'), 'wb',
    ) as f:
        pickle.dump(Gs, f)
#%%
