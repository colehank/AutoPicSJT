from __future__ import annotations

import json
import os
import time
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import APIConnectionError
from openai import BadRequestError
from openai import OpenAI

from ..config import LLMConfig
from ..utils import llm_utils
load_dotenv()


class BaseLLM:
    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv('LLM_URL'),
            api_key=os.getenv('LLM_API'),
        )
        self.model = LLMConfig.model
        self.max_tokens = LLMConfig.max_tokens
        self.temperature = LLMConfig.temperature
        self.top_p = LLMConfig.top_p
        self.top_k = LLMConfig.top_k
        self.frequency_penalty = LLMConfig.frequency_penalty
        self.presence_penalty = LLMConfig.presence_penalty
        self.stop = LLMConfig.stop
        self.response_format = LLMConfig.response_format

    def _repr_html_(self, title='LLM Settings'):
        params = {
            'model': self.model,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty,
            'stop': self.stop,
            'response_format': self.response_format,
        }
        df = pd.DataFrame(params.items(), columns=['Parameter', 'Value'])
        df_html = df.to_html(index=False)
        df_html = f"""
            <h3 style="color: #9FE2BF;">{title}</h3>
            {df_html}
        """
        return df_html

    def call(
        self,
        msg: list[dict],
        json=True,
        retries=3,
    ) -> dict[Any, Any] | str:
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    messages=msg,
                    response_format=self.response_format,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    stop=self.stop,
                )
                break
            except (
                BadRequestError, APIConnectionError,
            ) as err:
                if attempt < retries - 1:
                    time.sleep(1)
                else:
                    print(f'after {attempt + 1} attempts, failed to call LLM')
                    raise err
        content = response.choices[0].message.content.strip()

        if json:
            extracted_json = llm_utils.extract_json(content)
            if extracted_json is None:
                raise ValueError(
                    'Failed to extract JSON from the response content',
                )
            return extracted_json
        else:
            return content
