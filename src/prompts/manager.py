from __future__ import annotations

import importlib.util
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from string import Template
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

class PromptTemplateManager:
    """
    自动管理各种 NLP 任务的 prompt 模板。
    该实现会自动扫描当前目录（或者指定目录）中包含 “prompt” 的 Python 文件，
    加载其中的 prompt_template，并通过扫描模板内容中以 $ 开头的变量自动提取 required_params。
    """

    def __init__(self, templates_dir: str | None = None):
        """
        如果未指定模板目录，则默认使用当前文件所在目录。
        """
        if templates_dir is None:
            templates_dir = Path(__file__).parent
        else:
            templates_dir = Path(templates_dir)
        self._templates = self._load_templates(templates_dir)

    def _load_templates(self, directory: Path) -> dict[str, dict]:
        """
        动态加载指定目录及其子目录下所有包含 prompt_template 定义的 Python 文件，并提取所需参数。

        递归遍历目录及其子目录中所有以 .py 结尾且文件名中包含 "prompt" 的文件，
        利用 importlib 加载模块，寻找模块内定义的 prompt_template，
        并通过正则表达式扫描角色为 'user' 的消息中 $xxx 的占位符，构建 required_params 列表。

        如果模板在子目录中，则模板的键名为"子目录名/文件名"，例如"cues_enrich/emotion_analysis_prompt"。

        Args:
            directory: 要扫描的目录路径

        Returns:
            一个字典，键为模板所在文件（模块）的名称（包含目录路径），值为包含以下两个键的字典：
                - 'template': prompt_template 内容（列表格式）
                - 'required_params': 自动提取出来的所需参数列表
        """
        templates = {}

        def scan_directory(dir_path: Path, relative_path: str = ''):
            for item in dir_path.iterdir():
                if item.is_file() and item.suffix == '.py' and 'prompt' in item.stem:
                    spec = importlib.util.spec_from_file_location(item.stem, item)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, 'prompt_template'):
                        prompt_template = getattr(module, 'prompt_template')
                        required_params = set()
                        # 遍历模板列表中所有角色为 'user' 的消息，查找 "$变量" 占位符
                        for message in prompt_template:
                            if message.get('role') == 'user':
                                found = re.findall(r'\$(\w+)', message.get('content', ''))
                                required_params.update(found)

                        # 构建包含目录结构的模板名称
                        template_key = f'{relative_path}/{item.stem}' if relative_path else item.stem
                        template_key = template_key.lstrip('/')

                        templates[template_key] = {
                            'template': prompt_template,
                            'required_params': list(required_params),
                        }
                elif item.is_dir():
                    # 递归扫描子目录，并保持目录路径信息
                    new_relative_path = f'{relative_path}/{item.name}' if relative_path else item.name
                    scan_directory(item, new_relative_path)

        # 开始递归扫描目录
        scan_directory(directory)
        return templates

    def get_template(self, task_name: str) -> list[dict[str, str]] | None:
        """
        根据任务名称获取对应的 prompt 模板。

        Args:
            task_name: 模板名称（一般对应文件名）

        Returns:
            模板内容的列表，若没有找到则返回 None
        """
        task_info = self._templates.get(task_name)
        if task_info:
            return task_info.get('template')
        return None

    def make_prompt(self, task_name: str, passage: str, **kwargs) -> list[dict[str, str]]:
        """
        根据指定任务名称和参数创建 prompt。

        Args:
            task_name: 要使用的 prompt 模板名称（对应加载的文件名）
            passage: 文章或文本段
            **kwargs: 模板所需的其他参数（会自动验证是否满足模板中通过 "$" 定义的 required_params）

        Returns:
            一个 message 字典列表，构成最终的 prompt 内容

        Raises:
            ValueError: 当 task_name 无效或所需参数缺失时触发
        """
        if task_name not in self._templates:
            available = list(self._templates.keys())
            raise ValueError(f'无效的 task_name: {task_name}。可用的任务有: {available}')

        template_info = self._templates[task_name]
        # 构造参数字典：必须包含 passage，再加上其它传入参数
        params = {'passage': passage, **kwargs}

        # 检查是否所有必需参数都已提供
        missing_params = [
            param for param in template_info['required_params'] if param not in params
        ]
        if missing_params:
            raise ValueError(f'{task_name} 模板缺失所需参数: {missing_params}')

        # 如果某些参数为列表，则转换为 JSON 字符串（例如 named_entities、word 等）
        for key in params:
            if isinstance(params[key], list):
                params[key] = json.dumps(params[key], ensure_ascii=False)

        return self._process(template_info['template'], **params)

    def _process(self, prompt_template: list[dict[str, str]], **kwargs) -> list[dict[str, str]]:
        """
        进行模板替换，将模板中的 "$变量" 替换为实际值。

        Args:
            prompt_template: 模板内容，列表格式
            **kwargs: 替换模板中变量的对应参数

        Returns:
            替换后的模板内容列表

        Raises:
            ValueError: 如果模板中引用的变量缺失
        """
        processed = deepcopy(prompt_template)
        for item in processed:
            if item.get('role') == 'user':
                try:
                    template = Template(item['content'])
                    item['content'] = template.substitute(**kwargs)
                except KeyError as e:
                    raise ValueError(f'模板缺失参数: {e}')
        return processed
