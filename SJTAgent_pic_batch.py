# %%
from __future__ import annotations

from PIL import Image
from src_img import PicSJTAgent
from src_img.run import SJTRunner
from src_text import DataLoader
# %%
ref_name: str = "Ye"
ref_img_path: str = "resources/ref_character/male.png"
model: str = "gpt-5" # 建议使用 gpt-5，目前只有gpt-5能良好完成每个流程的任务
verbose: bool = True
res_dir: str = "outputs_2"
# %%
dl = DataLoader()
sjts = dl.load("PSJT-Mussel", "en")
neopir_meta = dl.load_meta("NEO-PI-R")
sjts_dims = ['N4', 'E2', 'O5', 'A4', 'C5']
full_dims = [f"{neopir_meta[dim]['domain']}: {neopir_meta[dim]['facet_name']}" for dim in sjts_dims]
trait_label_map = {k:v for k,v in zip(sjts_dims, full_dims)}
ref_img = Image.open(ref_img_path)
# %%
all_tasks = [
    {
        "situ": sjts[trait][item_id],
        "trait": trait_label_map[trait],
        "fname": f"{trait}_{item_id}"
    }
    for trait in sjts_dims
    for item_id in sjts[trait]
]
# %%
runner = SJTRunner(
    tasks=all_tasks,
    ref_name=ref_name,
    ref_img=ref_img,
    model=model,
    output_dir=res_dir,
)
runner.cook_async(skip_generated=True, continue_on_error=True)
# %%
all_data = {}
import os
import json
import os.path as op
items = [op.join(res_dir, f) for f in os.listdir(res_dir) if f.endswith(".json")]

for item in items:
    with open(item, "r") as f:
        data = json.load(f)
    all_data[op.basename(item).replace(".json", "")] = data
with open(op.join(res_dir, "all_data.json"), "w") as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)
#%%
