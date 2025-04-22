# %%
from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor

from tqdm.autonotebook import tqdm
from wasabi import msg

import src
from src.pipeline import SituationProcessor

results_dir = 'results/final'
traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
n_item = 21
max_workers = len(traits) * n_item

tasks = [(trait, str(item_id)) for trait in traits for item_id in range(n_item + 1)]
dm = src.DataManager()
situs = dm.read('situation_judgment_test', 'SJTs', extract_stiu=True)

failed_tasks = []

# %%
def task(trait, item_id, model='gpt-4o', ref_you='Ye', max_attempts=10):
    situ = situs[trait[0]][item_id]
    attempts = 0
    while attempts < max_attempts:
        try:
            P = src.SituationProcessor(
                model=model,
                situ=situ,
                trait=trait,
                ref=ref_you,
            )
            res = P.fit(verbose=False)
            return trait, item_id, res
        except Exception as e:
            attempts += 1
            tqdm.write(f'[RETRY {attempts}/{max_attempts}] Error for {trait}-{item_id}: {e}')
            if attempts >= max_attempts:
                tqdm.write(f'[FAILED] Maximum retry attempts reached for {trait}-{item_id}')
                failed_tasks.append((trait, item_id))
                return trait, item_id, None

all_results = {trait: {} for trait in traits}
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(task, trait, item_id): (trait, item_id) for trait, item_id in tasks}
    for future in tqdm(as_completed(futures), total=len(futures), desc='Processing all'):
        trait, item_id, res = future.result()
        all_results[trait][item_id] = res
#%%
for trait, items in tqdm(all_results.items(), desc='Processing traits'):
    this_dir = f'{results_dir}/{trait[0]}'
    fig_dir = f'{this_dir}/figs'
    data_dir = f'{this_dir}/data'
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    for item_id, res in tqdm(items.items(), desc=f'Saving {trait} items', leave=False):
        if res is None:
            tqdm.write(f'Skipping {trait}_{item_id} (result is None)')
            continue
        fig_G = src.draw_G(res['situation_graph'])
        fig_Gs = src.draw_Gs(res['vng_graphs'])
        fig_intergarted_Gs = src.draw_Gs(res['intergrated_Gs'])
        fig_G.savefig(f'{fig_dir}/G_{trait[0]}_{item_id}.tif', dpi=300, bbox_inches='tight')
        fig_Gs.savefig(f'{fig_dir}/Gs_{trait[0]}_{item_id}.tif', dpi=300, bbox_inches='tight')
        fig_intergarted_Gs.savefig(f'{fig_dir}/GsEnriched_{trait[0]}_{item_id}.tif', dpi=300, bbox_inches='tight')

        with open(f'{data_dir}/{trait[0]}_{item_id}_all.pkl', 'wb') as f:
            pickle.dump(res, f)

        for re in res:
            if re in ['cues', 'enriched_cues', 'Gs_prompt', 'Gs_prompt_polished']:
                with open(f'{data_dir}/{trait[0]}_{item_id}_{re}.json', 'w') as f:
                    json.dump(res[re], f, indent=4)
