# %%
import json
import os.path as op
from collections import defaultdict
from typing import Dict, Literal

from PIL import Image

from src_img import PicSJTAgent
from src_text import DataLoader

MODELS = ['gpt-5', 'deepseek-v3', 'gpt-4', 'gpt-3.5-turbo', 'o4-mini']
USER_SELECTABLE_REF_VIZ = ['build-in', 'description', 'upload']
class PicSJTSingleRunner:
    """Facade around :class:`PicSJTAgent` for single SJT item generation."""
    def __init__(
        self,
        ref_name: str = 'Ye',
        ref_viz_dir: str | None = None,
        output_dir: str = 'outputs',
        dataset: str = 'PSJT-Mussel',
        language: Literal['zh', 'en'] = 'zh',
    ) -> None:
        self.ref_name = ref_name
        self.ref_viz_dir = ref_viz_dir or op.join('resources', 'ref_character')
        self.output_dir = output_dir
        self.dataset = dataset
        self.language = language

        self.dataloader = DataLoader()
        self._neo_pi_r_meta = self.dataloader.load_meta('NEO-PI-R')
        self.sjts = self.dataloader.load(dataset, language)

        self.available_traits = list(self.sjts.keys())
        self.traits_to_select = self._build_traits_lookup()
        self.available_ref_viz = self._build_available_ref_viz()

    def _build_traits_lookup(self) -> Dict[str, Dict[str, str]]:
        traits_map: Dict[str, Dict[str, str]] = defaultdict(dict)
        for trait_code, meta in self._neo_pi_r_meta.items():
            if trait_code in self.available_traits:
                traits_map[meta['domain']][meta['facet_name']] = trait_code
        return traits_map

    def _build_available_ref_viz(self) -> Dict[str, str]:
        return {
            'male': op.join(self.ref_viz_dir, 'male.png'),
            'female': op.join(self.ref_viz_dir, 'female.png'),
        }

    def _resolve_trait_code(self, domain: str, facet: str) -> str:
        try:
            facet_map = self.traits_to_select[domain]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise KeyError(f"Domain '{domain}' is unavailable. Choose from {list(self.traits_to_select)}") from exc

        try:
            return facet_map[facet]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise KeyError(
                f"Facet '{facet}' is not mapped under domain '{domain}'. Choose from {list(facet_map)}"
            ) from exc

    def get_trait_code(self, domain: str, facet: str) -> str:
        return self._resolve_trait_code(domain, facet)

    def get_ref_viz(
        self,
        user_choice: Literal['build-in', 'description', 'upload'],
        *,
        build_in_choice: str = '',
        description: str = '',
        upload_path: str = '',
    ) -> str | Image.Image:
        if user_choice == 'build-in':
            if build_in_choice not in self.available_ref_viz:
                raise ValueError(
                    f"Only {list(self.available_ref_viz.keys())} are supported for build-in reference visualization."
                )
            with Image.open(self.available_ref_viz[build_in_choice]) as img:
                return img.copy()

        if user_choice == 'description':
            if not description:
                raise ValueError('Description text must be provided when using the description reference type.')
            return description

        if user_choice == 'upload':
            if not upload_path:
                raise ValueError('Upload path must be provided when using the upload reference type.')
            upload_abs_path = op.abspath(upload_path)
            with Image.open(upload_abs_path) as img:
                return img.copy()

        raise ValueError(f"Invalid user_choice: {user_choice}")

    def run_single(
        self,
        *,
        domain: str,
        facet: str,
        sjt_index: str,
        model: str,
        ref_choice: Literal['build-in', 'description', 'upload'],
        build_in_choice: str = '',
        description: str = '',
        upload_path: str = '',
    ) -> dict:
        if model not in MODELS:
            raise ValueError(f"Model '{model}' is unavailable. Choose from {MODELS}.")

        trait_code = self._resolve_trait_code(domain, facet)
        sjt_item = self.sjts[trait_code][sjt_index]
        trait_label = f"{domain}: {facet}"

        ref_viz = self.get_ref_viz(
            user_choice=ref_choice,
            build_in_choice=build_in_choice,
            description=description,
            upload_path=upload_path,
        )

        runner = PicSJTAgent(
            situ=sjt_item,
            trait=trait_label,
            ref_name=self.ref_name,
            ref_viz=ref_viz,
            model=model,
            output_dir=self.output_dir,
            output_fname=f"{trait_code}_{sjt_index}",
        )
        return runner.run()

    def load_generation_artifacts(self, trait_code: str, sjt_index: str) -> dict:
        output_path = op.join(self.output_dir, f"{trait_code}_{sjt_index}.json")
        with open(output_path, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    
    def trait_code_to_domain_facet(self, trait_code: str) -> tuple[str, str]:
        meta = self._neo_pi_r_meta[trait_code]
        return meta['domain'], meta['facet_name']
# %%
if __name__ == '__main__':
    runner = PicSJTSingleRunner(language='en')
    domain_choice = 'Neuroticism' # 'Extraversion' | 'Agreeableness' | 'Conscientiousness' | 'Openness' | 'Neuroticism'
    facet_choice = 'Self-consciousness'
    
    trait_code = 'A4'
    domain_choice, facet_choice = runner.trait_code_to_domain_facet(trait_code)
    sjt_idx = '11'
    model_choice = 'gpt-5'
    ref_choice = 'build-in'
    build_in_choice = 'male'
# %%
    runner.run_single(
        domain=domain_choice,
        facet=facet_choice,
        sjt_index=sjt_idx,
        model=model_choice,
        ref_choice=ref_choice,
        build_in_choice=build_in_choice,
    )
# %%
    trait_code = runner.get_trait_code(domain_choice, facet_choice)
    artifacts = runner.load_generation_artifacts(trait_code, sjt_idx)
    image_info = artifacts['situation']
    options_info = artifacts['options']


#%%
