# %%
from __future__ import annotations

import os
import os.path as op

from PIL import Image
from PIL import ImageOps

from src.utils.image_utils import make_sequence

def load_vng_pics(trait, itemID, res_dir):
    des_dir = f'{res_dir}/{trait}/{trait}_{itemID}'
    fs = os.listdir(des_dir)
    pics = [i for i in fs if i.endswith('.png')]
    vng_order = ['E', 'I', 'Pr', 'P']
    vng_pics = {}
    for vng_type in vng_order:
        for pic in pics:
            if pic.startswith(vng_type + '.'):
                vng_pics[vng_type] = Image.open(op.join(des_dir, pic))
                break

    return vng_pics
#%%
pcs = load_vng_pics('N', '1', 'results/final/output')
cms = make_sequence(list(pcs.values()))
cms
#%%
for i in range(21):
    pcs = load_vng_pics('N', str(i), 'results/final/output')
    cms = make_sequence(list(pcs.values()))
    cms.save(f'vngs/N_{i}.png')
