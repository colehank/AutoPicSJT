from __future__ import annotations

from itertools import chain

import networkx as nx
import pandas as pd
from tqdm.autonotebook import tqdm

from ..models.llms import TempletLLM
from ..utils.graph_utils import build_G
from ..utils.graph_utils import dic_G
from ..utils.graph_utils import extract_knowledge
from ..utils.graph_utils import find_node_by_value
from ..utils.graph_utils import get_max_attribute
from .cues_enrich import enrich_characters
from .cues_enrich import enrich_objects
from .cues_enrich import enrich_scenes
from .utils import _replace_pronouns
from .utils import find_key_in_result
from .utils import identify_cue_type


class SituationProcessor:
    """A processor for generating and analyzing situation graphs."""
    # ✅
    def __init__(self, situ, trait, model='gpt-4o', ref = 'Ye', debug=False):
        """Initialize the processor with a specific model."""
        self.llms = {
            'sg': TempletLLM('sg_generation'),
            'vng': TempletLLM('vng_from_graph'),
            'cue_ext':  TempletLLM('cues_extraction'),
            'cls_node': TempletLLM('classfy_node'),
            'G2str': TempletLLM('graph2prompt'),
            'vng_polisher': TempletLLM('vng_polisher'),
        }
        self.cue_types = ['att|obj', 'obj-obj', 'att|obj-obj', 'obj-att|obj', 'att|obj-att|obj']
        self.debug = debug
        self.ref = ref
        self.situ = _replace_pronouns(situ, ref)
        self.trait = trait

        for _, llm in self.llms.items():
            llm.model = model
    # ✅
    def situ_graph(self):
        """Generate a situation graph based on the given situation."""
        if self.debug:
            self.G = 'nx.Graph'
            return 'nx.Graph'

        res = self.llms['sg'].call(self.situ)
        res_sg = find_key_in_result(res, 'SceneGraph')['SceneGraph']
        G = build_G(res_sg)

        self.G:nx.Graph = G

    # ✅
    def Gs_from_situ(self):
        """Design visual narrative graphs based on the situation and situation graph."""
        if self.debug:
            self.Gs = 'dict[str, nx.Graph]'
            return 'dict[str, nx.Graph]'
        str_G = dic_G(self.G)
        res = self.llms['vng'].call(self.situ, graph=str_G)
        res_vng = find_key_in_result(res, 'VNG')['VNG']
        res_Gs = {
            vng: build_G(content) for vng, content in res_vng.items()
        }

        self.Gs:dict[str, nx.Graph] = res_Gs

    # ✅
    def extract_cues_from_Gs(self):
        """对每个VNG进行它的cue的提取,返回一个字典, key为VNG的id, value为提取的cues
        这里的cues是"""
        if self.debug:
            self.cues = 'dict[str, list[dict[str, list[str]]]]'
            return 'dict[str, list[dict[str, list[str]]]]'

        Gs_klg = {vng_idx: self._get_knowledge(G) for vng_idx, G in self.Gs.items()}
        res = self.llms['cue_ext'].call(
            self.situ, trait=self.trait, graphs = Gs_klg,
        )
        cues = find_key_in_result(res, 'cues')['cues']
        self.cues:dict[str:list[dict[str, list[str]]]] = cues

    # ✅
    def enrich_Gs_by_cues(self):
        if self.debug:
            self.enriched_Gs = 'dict[str, nx.Graph]'
            return 'dict[str, nx.Graph] -> each VNG_G is enriched by its cues'
        enriched_Gs_cues = {}
        for vng_idx, cues in self.cues.items():
            if cues == []:
                continue
            else:
                enriched_Gs_cues[vng_idx] = self._enrich_G_cues(
                situ = self.situ, trait=self.trait, cues=cues, ref=self.ref,
                )

        self.enriched_Gs_cues:dict[str, dict] = enriched_Gs_cues
        self.enriched_Gs:dict[str, nx.Graph] = self._add_cues_to_Gs(
            enriched_Gs_cues=enriched_Gs_cues,
            Gs=self.Gs,
            G=self.G,
        )

    # ✅
    def intergrate_enriched_Gs(self) -> dict[str, nx.Graph]:
        if self.debug:
            self.intergerated_Gs = 'dict[str, nx.Graph]'
            return 'dict[str, nx.Graph]'

        intergerated_Gs = {}
        for vng_idx, G in self.Gs.items():
            if vng_idx not in self.enriched_Gs:
                intergerated_Gs[vng_idx] = G
            else:
                intergerated_Gs[vng_idx] = self.enriched_Gs[vng_idx]

        self.intergerated_Gs:dict[str, nx.Graph] = intergerated_Gs

    def Gs2prompt(
        self,
        Gs:dict[str, nx.Graph] = None,
        size = '1024*1024',
        style = 'realistic',
    ):
        """Convert the graphs to string format."""
        Gs_str = {}
        for vng_idx, G in Gs.items():
            res = self.llms['G2str'].call(dic_G(G), size=size, style=style)
            res_str = find_key_in_result(res, 'prompt')['prompt']
            Gs_str[vng_idx] = res_str

        self.Gs_prompt: dict[str, str] = Gs_str

    def prompt_polish(self):
        res = self.llms['vng_polisher'].call(passage=self.situ, vng = self.Gs_prompt)
        res_str = find_key_in_result(res, 'VNG')['VNG']

        self.Gs_prompt_polished:dict[str, str] = res_str

    def fit(self, size = '1024x1024', style = 'realistic', verbose=False):
        """Fit the model to the situation and trait."""
        steps = [
            ('Generating situation graph', self.situ_graph),
            ('Creating visual narrative graphs', self.Gs_from_situ),
            ('Extracting cues from graphs', self.extract_cues_from_Gs),
            ('Enriching graphs with cues', self.enrich_Gs_by_cues),
            ('Integrating enriched graphs', self.intergrate_enriched_Gs),
            ('Converting graphs to prompt', lambda: self.Gs2prompt(self.intergerated_Gs, size, style)),
            ('Polishing the prompt', self.prompt_polish),
        ]

        if verbose:
            for desc, step_func in tqdm(steps, desc='Processing pipeline'):
                tqdm.write(f'Running: {desc}')
                step_func()
        else:
            for _, step_func in steps:
                step_func()

        return {
            'situation_graph': self.G,
            'vng_graphs': self.Gs,
            'cues': self.cues,
            'enriched_cues': self.enriched_Gs_cues,
            'enriched_Gs': self.enriched_Gs,
            'intergrated_Gs': self.intergerated_Gs,
            'Gs_prompt': self.Gs_prompt,
            'Gs_prompt_polished': self.Gs_prompt_polished,
        }

    def _enrich_G_cues(
        self,
        cues: list[dict[str, list[str]]],
        situ: str,
        trait: str,
        ref: str,
    ) -> dict[str, list[dict[str, list[str]]]]:
        """Enrich cues based on the situation graph and trait."""
        node_types = self._cls_cues_nodes(situ, cues)
        all_characters = [node_type.get('character', []) for node_type in node_types]
        all_scenes = [node_type.get('scene', []) for node_type in node_types]
        all_objects = [node_type.get('object', []) for node_type in node_types]

        uniuqe_characters = list(set(chain.from_iterable(all_characters)))
        uniuqe_scenes = list(set(chain.from_iterable(all_scenes)))
        uniuqe_objects = list(set(chain.from_iterable(all_objects)))

        if uniuqe_characters != []:
            enriched_characters = enrich_characters(
                situ, trait,
                ana_characters=uniuqe_characters,
                act_character=ref,
            )
        else:
            enriched_characters = {}

        if uniuqe_scenes != []:
            enriched_scenes = enrich_scenes(
                situ, trait,
                scenes=uniuqe_scenes,
                act_character=ref,
            )
        else:
            enriched_scenes = {}
        if uniuqe_objects != []:
            enriched_objects = enrich_objects(
                situ, trait,
                objects=uniuqe_objects,
                act_character=ref,
            )
        else:
            enriched_objects = {}

        enriched_cues = {
            'character': enriched_characters,
            'scene': enriched_scenes,
            'object': enriched_objects,
        }
        return enriched_cues

    def _add_cues_to_Gs(
        self,
        enriched_Gs_cues:dict[str, dict],
        Gs:dict[str, nx.Graph],
        G:nx.Graph,
    ) -> dict[str, nx.Graph]:
        enriched_Gs = {}
        for idx, G in Gs.items():
            if idx not in enriched_Gs_cues:
                continue

            new_G = G.copy()
            cues = enriched_Gs_cues[idx]

            # Define mapping for different cue types
            cue_configs = {
                'character': {'value_suffix': '_body', 'facial_key': 'facial'},
                'scene': {'value_suffix': '_scene'},
                'object': {'value_suffix': '_object'},
            }

            for cue_type, items in cues.items():
                if not items or cue_type not in cue_configs:
                    continue

                config = cue_configs[cue_type]

                for item_name, item_data in items.items():
                    node = find_node_by_value(G, item_name, 'object_node')
                    if not node:
                        continue

                    obj_num = node.split('_')[1]
                    max_attr_idx = get_max_attribute(G, node)

                    # Handle character's body and facial attributes
                    if cue_type == 'character':
                        # Add body attribute
                        body_attr = f'attribute|{obj_num}|{max_attr_idx+1}'
                        new_G.add_node(body_attr, type='attribute_node', value='_body', annot=item_data['body'])
                        new_G.add_edge(body_attr, node, type='attribute_edge')

                        # Add facial attribute
                        face_attr = f'attribute|{obj_num}|{max_attr_idx+2}'
                        new_G.add_node(face_attr, type='attribute_node', value='_face', annot=item_data['facial'])
                        new_G.add_edge(face_attr, node, type='attribute_edge')
                    else:
                        # Add single attribute for scene or object
                        attr = f'attribute|{obj_num}|{max_attr_idx+1}'
                        new_G.add_node(attr, type='attribute_node', value=config['value_suffix'], annot=item_data)
                        new_G.add_edge(attr, node, type='attribute_edge')

            enriched_Gs[idx] = new_G
        return enriched_Gs

    def _get_knowledge(self, G: nx.Graph) -> dict[str, list[str]]:
        """Extract knowledge patterns from the graph."""
        cues = {
            cue_type: extract_knowledge(G, cue_type=cue_type)
            for cue_type in self.cue_types
        }
        return cues

    def _cls_cue_nodes(self, situ: str, words: list[str]) -> dict[str, list[str]]:
        """Classify nodes based on the situation and words."""
        res = self.llms['cls_node'].call(situ, words=words)
        res_nodes = find_key_in_result(res, 'classification')['classification']
        return res_nodes

    def _cls_cues_nodes(self, situ: str, cues: list[dict[str, list[str]]]) -> list[dict[str, list[str]]]:
        """Classify cue nodes based on the situation and cues."""
        words = [identify_cue_type(cue)['nodes'] for cue in cues]
        clss = [self._cls_cue_nodes(situ, words) for words in words]
        return clss

    def _repr_html_(self):
        pipline = {
            'situation_graph': 'self.G' if hasattr(self, 'G') else 'not generated',
            'vng_graphs': 'self.Gs' if hasattr(self, 'Gs') else 'not generated',
            'cues': 'self.cues' if hasattr(self, 'cues') else 'not generated',
            'enriched_cues': 'self.enriched_Gs' if hasattr(self, 'enriched_Gs') else 'not generated',
            'intergrated_Gs': 'self.intergerated_Gs' if hasattr(self, 'intergerated_Gs') else 'not generated',
            'Gs_prompt': 'self.Gs_prompt' if hasattr(self, 'Gs_prompt') else 'not generated',
            'Gs_prompt_polished': 'self.Gs_prompt_polished' if hasattr(self, 'Gs_prompt_polished') else 'not generated',
        }

        df = pd.DataFrame(pipline.items(), columns=['Step', 'Result'])
        df_html = df.to_html(index=False)
        df_html = f"""
            <h3 style="color: #9FE2BF;">Pipeline Results</h3>
            {df_html}
        """
        return df_html
