import json
import os
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any
base_path = os.path.dirname(os.path.abspath(__file__))


@dataclass
class DatasetMetadata:
    description: str
    author: str
    source: str
    url: str
    data_source: str
    data: Dict[str, Dict[str, Any]] = field(default_factory=dict)

meta_SceneGraph = DatasetMetadata(
    description="A wide range of SceneGraph elements, including attributes, relations, objects, scene_attributes.",
    author="Ziqi Gao et., al.(2024)",
    source="arXiv:2412.08221 [cs.CV]",
    url="https://arxiv.org/abs/2412.08221",
    data_source="https://github.com/RAIVNLab/GenerateAnyScene/tree/main/metadata",
    data={
        "attributes": {
            "description": "A list of attributes describing objects.",
            "path": os.path.join(base_path, "any_scene_graph", "attributes.json")
        },
        "relations": {
            "description": "A list of relations between objects.",
            "path": os.path.join(base_path, "any_scene_graph", "relations.json")
        },
        "objects": {
            "description": "A list of objects in the scene.",
            "path": os.path.join(base_path, "any_scene_graph", "objects.json")
        },
        "scene_attributes": {
            "description": "A list of attributes describing the scene.",
            "path": os.path.join(base_path, "any_scene_graph", "scene_attributes.json")
        }
    }
)

situation_DIAMONDS = DatasetMetadata(
    description="A scale for evalute situation.",
    author="null",
    source="null",
    url="null",
    data_source="null",
    data={
        "DIAMONDS": {
            "description": "8 dims, 3 items per dim.",
            "path": os.path.join(base_path, "DIAMONDS.json")
        },
        "DIAMNODS_zh": {
            "description": "8 dims, 3 items per dim. Chinese version.",
            "path": os.path.join(base_path, "DIAMONDS_zh.json")
        }
    }
)

situation_judgment_test = DatasetMetadata(
    description="Situation judgement tests for personality.",
    author="null",
    source="null",
    url="null",
    data_source="null",
    data={
        "SJTs": {
            "description": "Situation judgement tests for personality.",
            "path": os.path.join(base_path, "SJTs.json")
        },
        "SJTs_zh": {
            "description": "Situation judgement tests for personality. Chinese version.",
            "path": os.path.join(base_path, "SJTs_zh.json")
        }
    }
)

class DataManager():
    def __init__(self):
        self.metadata = json.loads(json.dumps({
            'any_scene_graph': meta_SceneGraph.__dict__,
            'situation_DIAMONDS': situation_DIAMONDS.__dict__,
            'situation_judgment_test': situation_judgment_test.__dict__
        }, default=lambda o: o.__dict__))
    
    def _repr_html_(self):
        """
        Returns a string representation of the metadata in HTML format.
        """
        flattened_metadata = {
            key: {
                "data_head": key,
                "data_name": value.get("data", {}).keys(),
                "description": value.get("description", ""),
                "author": value.get("author", ""),
                "source": value.get("source", ""),
            }
            
            for key, value in self.metadata.items()
        }
        df = pd.DataFrame.from_dict(flattened_metadata, orient='index')
        return df.to_html(index = False, header=True, escape=True)
        
    
    def read(
        self, data_head, data_name, extract_stiu = False):
        
        if extract_stiu and data_head != 'situation_judgment_test':
            raise ValueError(f"Data head {data_head} does not support extraction.")
        
        if data_head not in self.metadata:
            raise ValueError(f"Data head {data_name} not found in metadata.")
        
        data_path = self.metadata[data_head]['data'][data_name]['path']
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if extract_stiu:
            situs = {
                triat:{
                    i:'.'.join(data[triat][i]['situ'].split('.')[:-1]) for i in data[triat]
                    } for triat in data}
            return situs
        else:
            return data
