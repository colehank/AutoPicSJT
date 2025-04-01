from typing import Dict, List, Any, Optional, Union, Callable
from .ner import prompt_template as ner_prompt_template
from .triple_extraction import prompt_template as triple_extraction_prompt_template
from .vng_generation import prompt_template as vng_generation_prompt_template
from .sg_generation import prompt_template as sg_generation_prompt_template
import json
from copy import deepcopy
from string import Template


class PromptTemplateManager:
    """Manages prompt templates for various NLP tasks."""
    
    def __init__(self):
        # Registry of available templates and their parameter requirements
        self._templates = {
            "NER": {"template": ner_prompt_template, "required_params": ["passage"]},
            "triple_extraction": {
                "template": triple_extraction_prompt_template, 
                "required_params": ["passage", "named_entities"]
            },
            "vng_generation": {"template": vng_generation_prompt_template, "required_params": ["passage"]},
            "sg_generation": {
                "template": sg_generation_prompt_template, 
                "required_params": ["passage"]
            }
        }
    
    def get_templete(self, task_name: str) -> Optional[List[Dict[str, str]]]:

        return self._templates.get(task_name, {}).get("template", None)

    def make_prompt(self, task_name: str, passage: str, **kwargs) -> List[Dict[str, str]]:
        """
        Creates a prompt for the specified task with the given parameters.
        
        Args:
            task_name: The type of prompt to create
            passage: The text passage to process
            **kwargs: Additional parameters required by specific templates
            
        Returns:
            A list of message dictionaries forming the complete prompt
            
        Raises:
            ValueError: If task_name is invalid or required parameters are missing
        """
        if task_name not in self._templates:
            raise ValueError(
                f"Invalid task name: {task_name}. Available tasks: {list(self._templates.keys())}")
        
        template_info = self._templates[task_name]
        params = {"passage": passage, **kwargs}
        
        # Validate required parameters are present
        missing_params = [param for param in template_info["required_params"] if param not in params]
        if missing_params:
            raise ValueError(f"Missing required parameters for {task_name}: {missing_params}")
            
        # Process named_entities if provided but as a list
        if "named_entities" in params and isinstance(params["named_entities"], list):
            params["named_entities"] = json.dumps(params["named_entities"], ensure_ascii=False)
            
        return self._process(template_info["template"], **params)

    def _process(self, prompt_template: List[Dict[str, str]], **kwargs) -> List[Dict[str, str]]:
        """
        Substitutes placeholders in prompt templates with actual values.
        
        Args:
            prompt_template: The template to process
            **kwargs: Parameters to substitute in the template
            
        Returns:
            The processed template with substitutions made
        """
        processed = deepcopy(prompt_template)
        
        for item in processed:
            if item['role'] == 'user':
                try:
                    template = Template(item['content'])
                    item['content'] = template.substitute(**kwargs)
                except KeyError as e:
                    raise ValueError(f"Missing parameter in template: {e}")
                
        return processed
