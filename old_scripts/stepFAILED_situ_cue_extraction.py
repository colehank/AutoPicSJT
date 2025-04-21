# %%
from __future__ import annotations

import json
import os
import os.path as op
import pickle
import re

import networkx as nx
import numpy as np
import pandas as pd
from joblib import delayed
from joblib import Parallel
from tqdm.autonotebook import tqdm
from tqdm_joblib import tqdm_joblib
from wasabi import msg

import src
from src import PromptTemplateManager
from src.models.llms import TempletLLM
from src.prompts.diamonds import prompt_template


result_dir = 'results/situ_character'
situs = src.DataManager().read('situation_judgment_test', 'SJTs')
# %%


def diamonds_score(data):
    means = {
        dim: np.mean([v for k, v in data.items() if k.startswith(dim)])
        for dim in ['D', 'I', 'A', 'M', 'O', 'N', 'Dc', 'S']
    }
    return means


def diamonds_to_bf(means):
    bf_map = {
        'O': ['I'], 'C': ['D'], 'E': ['O', 'S', 'M'], 'A': ['A', 'Dc'], 'N': ['N'],
    }
    return [np.mean([means[dm] for dm in v if dm in means]) for v in bf_map.values()]


def extract_attr_node(G):
    result = []
    for u, v in G.edges():
        if u.startswith('attribute') and v.startswith('object'):
            attribute_value = G.nodes[u].get('value', u)
            object_value = G.nodes[v].get('value', v)
            result.append(f'{object_value} is {attribute_value}')
    return result


def gen_situ_character(trait, item_id, results_root='results'):
    situ = situs[trait][item_id]
    p = f'{results_root}/event_graph/{trait}/{item_id}/{trait}_{item_id}.pkl'
    with open(p, 'rb') as f:
        res = pickle.load(f)
    G = res['G']
    cues = extract_attr_node(G)
    del res

    bf_res = {}
    dm_llm = TempletLLM('diamonds')
    dm_llm.llm.model = 'gpt-4o'
    for cue in cues:
        _res_dm = dm_llm.call(passage=situ, word=cue)
        res_dm = diamonds_score(_res_dm['DIAMONDS'])
        res_bf = diamonds_to_bf(res_dm)
        bf_res[cue] = res_bf

    columns = ['O', 'C', 'E', 'A', 'N']
    df = pd.DataFrame.from_dict(bf_res, orient='index', columns=columns)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'cue'}, inplace=True)
    return df


def merge_dfs(data_list):
    merged = {}
    for i, d in enumerate(data_list):
        for trait, q_dict in d.items():
            if trait not in merged:
                merged[trait] = {}
            for qnum, df in q_dict.items():
                if qnum not in merged[trait]:
                    merged[trait][qnum] = df[['cue']].copy()
                df_sel = df[['O', 'C', 'E', 'A', 'N']].copy()
                df_sel = df_sel.rename(
                    columns={col: f'{col}_{i}' for col in df_sel.columns},
                )
                merged[trait][qnum] = pd.concat(
                    [merged[trait][qnum], df_sel], axis=1,
                )

    for trait in merged:
        for qnum in merged[trait]:
            df = merged[trait][qnum]
            for factor in ['O', 'C', 'E', 'A', 'N']:
                cols = [
                    col for col in df.columns if col.startswith(factor + '_')
                ]
                if cols:
                    df[factor] = df[cols].mean(axis=1)
                    df[factor + '_std'] = df[cols].std(axis=1)
    return merged


def process_item(trait, item_id, results_root, n_repeat):
    """
    处理单个 item 的函数，内部实现重试逻辑。
    """
    while True:
        try:
            res = gen_situ_character(
                trait, str(
                    item_id,
                ), results_root=results_root,
            )
            return item_id, res
        except Exception as e:
            msg.fail(
                f'Failed to process {trait}_{item_id} after {n_repeat} attempts: {e}',
            )
            continue


def process_situ_character(traits, save=True, results_root='results', n_repeat=30, njobs=22):
    """
    对给定 traits 和 item 范围处理情景特征数据。

    Args:
        traits (list): 要处理的特征列表（例如 ['O', 'C', 'E', 'A', 'N']）。
        results_root (str): 存放结果的根目录。
        n_repeat (int): 重复次数。
        njobs (int): 并行计算时使用的进程数。

    Returns:
        dict: 处理后的情景特征数据。
    """
    output_dir = f'{results_root}/situ_character'
    os.makedirs(output_dir, exist_ok=True)
    all_res = []

    for i in tqdm(range(n_repeat), desc='Running repeats', position=0):
        if len(all_res) <= i:
            all_res.append({})
        for trait in tqdm(traits, desc='Running traits', position=1):
            if trait not in all_res[i]:
                all_res[i][trait] = {}
            with tqdm_joblib(desc=f'Running items for {trait}', total=22, position=2, leave=False):
                results = Parallel(n_jobs=njobs)(
                    delayed(process_item)(
                        trait, item_id, results_root, n_repeat,
                    )
                    for item_id in range(22)
                )
            for item_id, res in results:
                all_res[i][trait][item_id] = res

    # 假设 merge_dfs 是用于合并结果的函数
    all_res = merge_dfs(all_res)

    if save:
        os.makedirs(output_dir, exist_ok=True)
        pickle_path = op.join(output_dir, 'situ_character.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(all_res, f)
        msg.good(f'Results saved to {pickle_path}')
    else:
        return all_res

def process_data(data):
    """
    输入:
        data: 嵌套字典，外层键为特质，每个特质下是题目编号到 DataFrame 的字典
    输出:
        aggregated_df: 包含 cue、各特质的均值和标准差（聚合数据），以及 trait 和 question 列
        measurement_df: 包含 cue、各次测量数据（形如 O_0, C_1 等），以及 trait 和 question 列
    """
    # 1. 合并所有子 DataFrame，增加 trait 和 question 列
    dfs = []
    for trait, question_dict in data.items():
        for question, df in question_dict.items():
            df_temp = df.copy()
            df_temp['trait'] = trait
            df_temp['question'] = question
            dfs.append(df_temp)
    combined_df = pd.concat(dfs, ignore_index=True)

    # 2. 分离聚合数据与测量数据
    # 测量数据列名形如 "O_0", "C_1" 等
    measurement_cols = [col for col in combined_df.columns if re.match(r'.+_\d+$', col)]
    # 聚合数据为除了 "cue", "trait", "question" 和测量数据之外的其它列
    aggregated_other_cols = [
        col for col in combined_df.columns
        if col not in measurement_cols and col not in ['cue', 'trait', 'question']
    ]

    aggregated_df = combined_df[['cue'] + aggregated_other_cols + ['trait', 'question']]
    measurement_df = combined_df[['cue'] + measurement_cols + ['trait', 'question']]

    # 3. 对两个 DataFrame 的列进行重新排序，按特质顺序分组
    traits_order = ['O', 'C', 'E', 'A', 'N']

    # ----- 对 aggregated_df 排序 -----
    # 获取聚合数据中与特质相关的列（除去 "cue", "trait", "question"）
    agg_trait_cols = [col for col in aggregated_df.columns if col not in ['cue', 'trait', 'question']]
    new_agg_cols = ['cue']
    for trait in traits_order:
        # 如果存在均值列则加入
        if trait in agg_trait_cols:
            new_agg_cols.append(trait)
        # 如果存在标准差列则加入
        if f'{trait}_std' in agg_trait_cols:
            new_agg_cols.append(f'{trait}_std')
    # 若还有其他聚合数据列未被包含，则追加在后面
    remaining_agg = [col for col in agg_trait_cols if col not in new_agg_cols]
    new_agg_cols += remaining_agg
    new_agg_cols += ['trait', 'question']
    aggregated_df = aggregated_df[new_agg_cols]

    # ----- 对 measurement_df 排序 -----
    # 获取所有测量数据列（除去 "cue", "trait", "question"）
    meas_trait_cols = [col for col in measurement_df.columns if col not in ['cue', 'trait', 'question']]
    new_meas_cols = ['cue']
    for trait in traits_order:
        # 找出以该特质开头的测量列
        cols_for_trait = [col for col in meas_trait_cols if col.startswith(f'{trait}_')]
        # 按下划线后的数字排序
        cols_for_trait = sorted(cols_for_trait, key=lambda x: int(x.split('_')[1]))
        new_meas_cols += cols_for_trait
    new_meas_cols += ['trait', 'question']
    measurement_df = measurement_df[new_meas_cols]

    return aggregated_df, measurement_df


def highlight_situ_cues(
    G, situ_cues, highlight_color='green',
    results_dir='results/situ_character/figs',
    title='',
):
    os.makedirs(results_dir, exist_ok=True)
    situ_cues_nodeid = [
        (
            src.utils.get_node_id(G, i[0]),
            src.utils.get_node_id(G, i[1]),
        ) for i in situ_cues
    ]

    situ_cues_nodes = [item for tup in situ_cues_nodeid for item in tup]

    fig = src.draw_G(
        G,
        colors={
            'node': {
                node: ('skyblue', highlight_color) if node.startswith('object')
                else ('pink', highlight_color)
                for node in situ_cues_nodes
            },
        },
        title=title,
    )
    fig.savefig(
        f'{results_dir}/SituCues_{trait}_{item_id}.tif',
        dpi=300, bbox_inches='tight',
    )


# %%
if __name__ == '__main__':
    # process_situ_character(['O', 'C', 'E', 'A', 'N'], save=True, n_repeat=30)
    thre = 3  # threshold for cues defined as situation cues
    highlight = 'red'

    with open('results/situ_character/situ_character.pkl', 'rb') as f:
        all_res = pickle.load(f)

    final_data_avg, final_data_measure = process_data(all_res)
    final_data_avg.to_csv('results/situ_character/aggregated_data.csv', index=False)
    final_data_measure.to_csv('results/situ_character/measurement_data.csv', index=False)

    situ_cues_save = {}
    for trait in tqdm(['O', 'C', 'E', 'A', 'N'], desc='plotting traits', position=0):
        fig_results = f'results/situ_character/figs/{trait}'
        os.makedirs(fig_results, exist_ok=True)

        situ_cues_save[trait] = {}
        for item_id in tqdm(range(0, 22), desc=f'plotting items for {trait}', position=1, leave=False):
            note = f'*Trait-{trait} related cues - {trait}_{item_id} '

            with open(f'results/event_graph/{trait}/{item_id}/{trait}_{item_id}.pkl', 'rb') as f:
                G = pickle.load(f)['G']

            this = all_res[trait][item_id]
            situ_cues = this[this[trait] >= thre]['cue'].to_list()
            situ_cues = [i.split(' is ') for i in situ_cues]
            situ_cues_save[trait][item_id] = situ_cues
            highlight_situ_cues(
                G, situ_cues, highlight_color=highlight, results_dir=fig_results, title=note,
            )
    with open('results/situ_character/situ_cues.json', 'w') as f:
        json.dump(situ_cues_save, f, indent=4)

    # %%
    trait_columns = ['O', 'C', 'E', 'A', 'N']

    # 对每一行找出得分最高的特质
    aggregated_df = final_data_avg.copy()  # 避免修改原始数据
    aggregated_df['max_trait'] = aggregated_df[trait_columns].idxmax(axis=1)

    # 存储结果
    proportions = {}

    # 对每个特质计算比例
    for trait in trait_columns:
        # 选择 trait 列等于当前特质的行
        subset = aggregated_df[aggregated_df['trait'] == trait]
        if len(subset) > 0:
            # 计算比例：其中 max_trait 与当前特质一致的比例
            proportion = (subset['max_trait'] == trait).mean()  # True 计为1，False为0
            proportions[trait] = proportion
        else:
            proportions[trait] = None  # 如果没有对应的行，则设置为 None
