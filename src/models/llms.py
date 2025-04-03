from __future__ import annotations

import pandas as pd

from ..prompts import PromptTemplateManager
from ..utils.llm_utils import print_conversation
from .llm import BaseLLM


class TempletLLM():
    def __init__(self, task=None, json=True, **kwargs):
        self.prompt_manager = PromptTemplateManager()
        self.llm = BaseLLM()
        self.json = json
        self.task = task
        self.tasks = list(self.prompt_manager._templates.keys())

        if task is not None:
            self._check_task(task)
            self.template = self.prompt_manager.get_templete(task)

    def set_task(self, task):
        self. _check_task(task)
        self.task = task
        self.template = self.prompt_manager.get_templete(task)

    def call(self, passage, **kwargs):
        self.prompt = self.prompt_manager.make_prompt(
            self.task, passage, json=self.json, **kwargs,
        )
        return self.llm.call(self.prompt)

    def print(self, task=None):
        task = task or self.task
        if task is None:
            raise ValueError('No task specified. Please provide a task name.')
        self._check_task(task)
        template = self.prompt_manager.get_templete(task)
        print_conversation(template)

    def _repr_html_(self):
        if self.task is None:
            tasks = list(self.prompt_manager._templates.keys())
            params = {
                task: self.prompt_manager._templates[task]['required_params'] for task in tasks
            }
            df = pd.DataFrame(
                params.items(), columns=[
                    'Task', 'Required Parameters',
                ],
            )
            return df.to_html(index=False)
        else:
            return self.llm._repr_html_(f'TASK: {self.task}')

    def _check_task(self, task):
        if task not in self.prompt_manager._templates:
            raise ValueError(
                f'Invalid task name: {task}. Available tasks: {self.tasks}',
            )
