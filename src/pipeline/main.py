from __future__ import annotations

from itertools import chain

import networkx as nx

from ..models.llms import TempletLLM
from ..utils.graph_utils import build_G
from ..utils.graph_utils import dic_G
from ..utils.graph_utils import extract_knowledge
from ..viz.sg import draw_G
from ..viz.sg import draw_G_cue_highlight
from ..viz.sg import draw_Gs
from .cues_enrich import enrich_characters
from .cues_enrich import enrich_objects
from .cues_enrich import enrich_scenes
from .utils import _replace_pronouns
from .utils import find_key_in_result
from .utils import identify_cue_type
from .utils import which_vng_for_cues

class SituationGraphProcessor:
    """A processor for generating and analyzing situation graphs."""

    def __init__(self, situ, model='gpt-4o', ref = 'Ye'):
        """Initialize the processor with a specific model."""
        self.llms = {
            'sg': TempletLLM('sg_generation'),
            'vng': TempletLLM('vng_from_graph'),
            'cue_ext':  TempletLLM('trait_extraction'),
            'cue_enr': TempletLLM('cue_enrich'),
            'cls_node': TempletLLM('classfy_node'),
        }
        self.ref = ref
        self.situ = _replace_pronouns(situ, ref)

        for _, llm in self.llms.items():
            llm.model = model

    def situ_graph(self, situ: str) -> nx.Graph:
        """Generate a situation graph based on the given situation."""
        res = self.llms['sg'].call(situ)
        res_sg = find_key_in_result(res, 'SceneGraph')['SceneGraph']
        G = build_G(res_sg)
        return G

    def vng_design(self, situ: str, G: nx.Graph) -> dict[str, nx.Graph]:
        """Design visual narrative graphs based on the situation and situation graph."""
        str_G = dic_G(G)
        res = self.llms['vng'].call(situ, graph=str_G)
        res_vng = find_key_in_result(res, 'VNG')['VNG']
        res_Gs = {
            vng: build_G(content) for vng, content in res_vng.items()
        }
        return res_Gs

    def cues_extraction(self, situ: str, G: nx.Graph, trait: str) -> list[dict[str, list[str]]]:
        """Extract cues from the situation graph based on a specific trait."""
        pattern_G = self._get_knowledge(G)
        res = self.llms['cue_ext'].call(situ, trait=trait, graph=pattern_G)
        res_cues = find_key_in_result(res, 'cues')['cues']
        return res_cues

    def cues_enrich(self, situ: str, cues: list[dict[str, list[str]]], trait: str):
        """Enrich cues based on the situation graph and trait."""
        node_types = self._cls_cues_nodes(situ, cues)
        # Extract node types, providing empty lists for missing categories
        all_characters = [node_type.get('character', []) for node_type in node_types]
        all_scenes = [node_type.get('scene', []) for node_type in node_types]
        all_objects = [node_type.get('object', []) for node_type in node_types]

        uniuqe_characters = list(set(chain.from_iterable(all_characters)))
        uniuqe_scenes = list(set(chain.from_iterable(all_scenes)))
        uniuqe_objects = list(set(chain.from_iterable(all_objects)))

        enriched_characters = enrich_characters(
            situ, trait,
            ana_characters=uniuqe_characters,
            act_character=self.ref,
        )
        enriched_scenes = enrich_scenes(
            situ, trait,
            scenes=uniuqe_scenes,
            act_character=self.ref,
        )
        enriched_objects = enrich_objects(
            situ, trait,
            objects=uniuqe_objects,
            act_character=self.ref,
        )

        enriched_cues = {
            'character': enriched_characters,
            'scene': enriched_scenes,
            'object': enriched_objects,
        }
        return enriched_cues

    def vng_cue_mapping(
        self,
        vng_Gs: dict[str, nx.Graph],
        cues: list[dict[str, list[str]]],
        enriched_cues: dict[str, list[str]],
    ) -> dict[str, nx.Graph]:
        """Map cues to visual narrative graphs."""
        best_matchs = which_vng_for_cues(vng_Gs, cues)
        matched_Gs = [vng_Gs[idx] for idx in best_matchs]

        def which_VNG_to_add(vng_Gs, enriched_cues) -> list[str]:
            ...
        def add_enriched_cue_to_G(G, enriched_cues) -> nx.Graph:
            ...
        to_add_Gs = which_VNG_to_add(vng_Gs, enriched_cues)#@llm
        to_do_vng = [vng_Gs[idx] for idx in to_add_Gs]




    def fit(self, situ: str, trait: str):
        """Fit the model to the situation and trait."""

        G = self.situ_graph(situ)
        vng_Gs = self.vng_design(situ, G)
        cues = self.cues_extraction(situ, G, trait)
        enriched_cues = self.cues_enrich(situ, cues, trait)


        fig_G = draw_G(G)
        fig_Gs = draw_Gs(vng_Gs)
        fig_cue = draw_G_cue_highlight(G, cues)

        return {
            'situation': situ,
            'situation_graph': G,
            'vng_graphs': vng_Gs,
            'cues': cues,
            'enriched_cues': enriched_cues,
            'figs': {
                'situation_graph': fig_G,
                'vng_graphs': fig_Gs,
                'cues': fig_cue,
            },
        }

    def _get_knowledge(self, G: nx.Graph) -> dict[str, list[str]]:
        """Extract knowledge patterns from the graph."""
        cue_types = ['att|obj', 'obj-obj', 'att|obj-obj', 'obj-att|obj', 'att|obj-att|obj']
        cues = {
            cue_type: extract_knowledge(G, cue_type=cue_type)
            for cue_type in cue_types
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
