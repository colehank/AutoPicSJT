# %%
from __future__ import annotations

import base64
import os

import requests
from dotenv import load_dotenv
load_dotenv()


def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

name = 'https://www.dmxapi.cn/'
API_URL = name + 'v1/chat/completions'
API_KEY = os.getenv('LLM_API')
# %%
image_path = '/Users/zgh/Desktop/P-SJT/毕设/AutoPicSJT/ref_character/female_Ye_enhanced.png'  # <--------------------------------------------- 本地图片路径
base64_image = encode_image(image_path)
# %%
payload = {
    'model': 'o4-mini',
    'messages': [
        {
            'role': 'system',
            'content': [
                {
                    'type': 'text',
                    'text': "The upload image is Ye. User will ask you to generate image later. When the image generation prompt contains 'Ye', you should generate an image based on this character's visual feature as Ye's reference.",
                },
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/png;base64,{base64_image}'},
                },
            ],
        },
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': "Create a 1024x1024 realistic image of Ye on a tram with a friend, scene's description: The tram is moderately crowded, with people sitting and standing, ambient light filtering through the windows casting soft shadows, colors are muted with cool blues and grays, creating a calm atmosphere typical of a daily commute, Ye and his friend are casually engaged in conversation, the friend sitting next to Ye, as both appear relaxed and at ease."},
            ],
        },
    ],
    'user': 'DMXAPI',
}

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {API_KEY}',
    'User-Agent': f'DMXAPI/1.0.0 ({name})',
}

response = requests.post(API_URL, headers=headers, json=payload)
#%%
