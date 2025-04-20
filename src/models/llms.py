from __future__ import annotations

import pandas as pd

from ..prompts import PromptTemplateManager
from ..utils.llm_utils import print_conversation
from .llm import BaseLLM


class TempletLLM:
    def __init__(self, task: str = None, json: bool = True, **kwargs):
        self.prompt_manager = PromptTemplateManager()
        self.llm = BaseLLM()
        self.json = json
        self.task = task if task else None
        self.tasks = list(self.prompt_manager._templates.keys())
        self.prompts = []
        if task is not None:
            task = task.lower()
            self.task = f'{task}_prompt'
            self._check_task(self.task)
            self.template = self.prompt_manager.get_template(self.task)

    def set_task(self, task: str) -> None:
        self._check_task(task)
        self.task = task
        self.template = self.prompt_manager.get_template(task)

    def call(self, passage: str, **kwargs) -> dict:
        """
        根据指定任务及参数生成 prompt 并调用底层 LLM
        """
        self.prompt = self.prompt_manager.make_prompt(
            self.task, passage, **kwargs,
        )
        return self.llm.call(self.prompt, json=self.json)

    def print(self, task: str = None) -> None:
        """
        打印指定任务对应的 prompt 会话信息，若不指定 task，则使用当前任务
        """
        task = task or self.task
        if task is None:
            raise ValueError('未指定任务，请提供一个 task 名称。')
        self._check_task(task)
        template = self.prompt_manager.get_template(task)
        print_conversation(template)

    def _repr_html_(self) -> str:
        if self.task is None:
            tasks = list(self.prompt_manager._templates.keys())
            params = {
                task: self.prompt_manager._templates[task]['required_params'] for task in tasks
            }
            df = pd.DataFrame(
                list(params.items()),
                columns=['Task', 'Required Parameters'],
            )
            return df.to_html(index=False)
        else:
            return self.llm._repr_html_(f'TASK: {self.task}')

    def _check_task(self, task: str) -> None:
        if task not in self.prompt_manager._templates:
            raise ValueError(
                f'无效的 task 名称: {task}. 可用任务有: {self.tasks}',
            )
