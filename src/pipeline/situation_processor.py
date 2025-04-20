from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from wasabi import msg

import src
from src.models import TempletLLM
from src.utils import build_G
from src.utils import dic_G


def find_key_in_result(result: dict, target_key: str) -> dict:
    """
    在嵌套字典中查找特定键

    参数:
        result: 嵌套字典
        target_key: 要查找的键名

    返回:
        包含目标键的字典
    """
    current_dict = result

    while True:
        if target_key not in current_dict:
            if not any(isinstance(v, dict) for v in current_dict.values()):
                raise ValueError(f'无法找到{target_key}键')
            current_dict = next(v for v in current_dict.values() if isinstance(v, dict))
        else:
            return {target_key: current_dict[target_key]}

def cue_G_str(G):
    cue_types = ['att|obj', 'obj-obj', 'att|obj-obj', 'obj-att|obj', 'att|obj-att|obj']
    cues = {
        cue_type:src.extract_knowledge(
        G, cue_type=cue_type,
        ) for cue_type in cue_types
    }
    return cues


class SituationProcessor:
    """
    处理单个情景的类,整合了事件图提取、线索提取和故事板提取功能

    属性:
        situ (str): 情景文本
        trait (str): 性格特质,如 'O', 'C', 'E', 'A', 'N'
    """

    # 性格特质映射


    def __init__(self, situ, trait, model: str = 'claude-3-5-sonnet-latest'):
        BF_MAP = {
        'O': 'Openness',
        'C': 'Conscientiousness',
        'E': 'Extraversion',
        'A': 'Agreeableness',
        'N': 'Neuroticism',
        }
        self.situ = situ
        self.trait = BF_MAP[trait]

        # 初始化LLM模型
        self.sg_llm = TempletLLM('sg_generation')
        self.sg_llm.llm.model = model
        self.sg_llm.llm.top_k = 1
        self.vng_llm = TempletLLM('vng_generation')
        self.vng_llm.llm.model = model
        self.vng_llm.llm.top_k = 1
        self.cues_extractor = TempletLLM('trait_extraction')
        self.cues_extractor.llm.model = model
        self.cues_extractor.llm.top_k = 1
        self.vng_maker = TempletLLM('vng_from_graph')
        self.vng_maker.llm.model = model
        self.vng_maker.llm.top_k = 1

        self.original_results = {}

    def _retry_call(self, func, *args, max_attempts: int = 10000, **kwargs):
        """统一的重试工具"""
        for attempt in range(max_attempts):
            try:
                result = func(*args, **kwargs)
                if result is not None:
                    return result
                msg.warn(f'Attempt {attempt+1}/{max_attempts} failed, retrying...')
            except Exception as e:
                msg.warn(f'Error in attempt {attempt+1}/{max_attempts}: {e}')
                if attempt == max_attempts - 1:
                    raise
        raise RuntimeError(f'Failed after {max_attempts} attempts')

    def extract_event_graph(self) -> Any:
        """
        提取事件图

        返回:
            Any: networkx.DiGraph 格式的事件图
        """
        # 提取场景图
        result = self._retry_call(self.sg_llm.call, self.situ)
        res_sg = find_key_in_result(result, 'SceneGraph')
        self.G = build_G(res_sg['SceneGraph'])
        return self.G

    def extract_cues(self, G = None) -> list[dict]:
        """
        提取线索

        参数:
            G: 事件图,如果为None则会先提取事件图

        返回:
            List[Dict]: 提取的线索列表
        """
        if G is None:
            G = self.G
        str_G = cue_G_str(G)
        # 提取线索
        result = self._retry_call(
            self.cues_extractor.call,
            passage=self.situ,
            trait=self.trait,
            graph=str_G,
        )
        cues = find_key_in_result(result, 'cues')['cues']

        return cues

    def extract_storyboard(self, G = None) -> dict[str, Any]:
        """
        提取故事板

        参数:
            G: 事件图,如果为None则会先提取事件图

        返回:
            Dict[str, Any]: 故事板图字典
        """
        if G is None:
            G = self.G

        # 转换图为字典格式
        dic_G_data = dic_G(G)

        # 提取故事板
        result = self._retry_call(
            self.vng_maker.call,
            passage=self.situ,
            graph=dic_G_data,
        )
        res = find_key_in_result(result, 'VNG')

        # 构建VNG图
        vng_Gs = {idx: build_G(dat) for idx, dat in res['VNG'].items()}

        return vng_Gs

    def run(self) -> dict[str, Any]:
        """
        运行完整的处理流程

        返回:
            Dict[str, Any]: 包含事件图(G)、线索(cues)和故事板(vng_Gs)的字典
        """
        # 1. 提取事件图
        G = self.extract_event_graph()

        # 2. 提取线索
        cues = self.extract_cues(G)

        # 3. 提取故事板
        vng_Gs = self.extract_storyboard(G)

        return {
            'G': G,
            'cues': cues,
            'vng_Gs': vng_Gs,
        }
