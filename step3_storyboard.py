# %%
from __future__ import annotations

import os
import os.path as op
import pickle
import time
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import networkx as nx
from tqdm.autonotebook import tqdm
from wasabi import msg

import src
from src import TempletLLM
# %%
dm = src.DataManager()
kg_corrector = TempletLLM('kg_correct')

situs = dm.read('situation_judgment_test', 'SJTs', extract_stiu=True, replace_you=True)
Gs = dm.read('situation_judgement_test_G', 'G')
#%%
def pipeline(trait, item_id):
    # Load the situation and graph
    situ = situs[trait][str(item_id)]
    G = Gs[f'{trait}_{item_id}']

    # Convert graph to dictionary format
    dic_G = src.dic_G(G)

    # Initialize the VNG maker
    vng_maker = TempletLLM('vng_from_graph')

    # Call the VNG maker with the situation and graph
    res = vng_maker.call(passage=situ, graph=dic_G)

    # Build VNG graphs from the result
    vng_Gs = {idx: src.build_G(dat) for idx, dat in res['VNG'].items()}

    return vng_Gs

def run_with_retries(trait, item_id, max_retries=3, delay=.5):
    """
    尝试执行 pipeline，如果失败则重试。

    参数：
      trait (str): 任务的 trait 标识
      item_id (int): 任务的 item id
      max_retries (int): 最大重试次数，默认3次
      delay (float): 每次重试前等待的秒数，默认1秒

    返回：
      pipeline 函数的执行结果

    如果所有重试均失败，则抛出异常。
    """
    for attempt in range(max_retries):
        try:
            return pipeline(trait, item_id)
        except Exception as e:
            msg.warn(f'任务 {trait}_{item_id} 第 {attempt+1} 次执行失败: {e}')
            if attempt < max_retries - 1:
                time.sleep(delay)  # 等待指定时间后重试
            else:
                msg.fail(f'任务 {trait}_{item_id} 重试 {max_retries} 次后依然失败。')
                raise
# %%
# 构建任务列表
tasks = [(trait, item_id) for trait in ['O', 'C', 'E', 'A', 'N'] for item_id in range(22)]
vng_Gs = {}

# 使用 ThreadPoolExecutor 并行处理任务，结合 tqdm 显示进度
with ThreadPoolExecutor() as executor:
    future_to_task = {
        executor.submit(run_with_retries, trait, item_id): (trait, item_id)
        for trait, item_id in tasks
    }

    for future in tqdm(as_completed(future_to_task), total=len(future_to_task)):
        trait, item_id = future_to_task[future]
        try:
            result = future.result()
            # 存入结果字典
            vng_Gs[f'{trait}_{item_id}'] = result
        except Exception as e:
            msg.fail(f'任务 {trait}_{item_id} 出现异常：{e}')

with open('results/vng_graph/vng_Gs.pkl', 'wb') as f:
    pickle.dump(vng_Gs, f)
#%%
tasks = [(trait, item_id) for trait in ['O', 'C', 'E', 'A', 'N'] for item_id in range(22)]
with open('results/vng_graph/vng_Gs.pkl', 'rb') as f:
    vng_Gs = pickle.load(f)


plt.rcParams['font.family'] = 'Times New Roman'

res_root = 'results/vng_graph'
os.makedirs(res_root, exist_ok=True)

for task in tasks:
    trait, item_id = task
    note = f"*PersonalitySJT's VNG - {trait}_{item_id} "
    print(f'\r{note} ', flush=True, end='')
    result_path = op.join(
        res_root, trait,
        f'{trait}_{item_id}.tif',
    )
    os.makedirs(op.dirname(result_path), exist_ok=True)
    fig = src.draw_Gs(vng_Gs[f'{trait}_{item_id}'])
    # fig.text(0.5, 0, note, fontsize=10, ha='center', va='bottom')
    fig.savefig(
        result_path,
        dpi=300, bbox_inches='tight',
    )
    plt.close('all')
