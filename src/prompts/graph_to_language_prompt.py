from __future__ import annotations

from string import Template

condition_system = """
# Graph to Language Expert
Convert the given knowledge graph to natural language.

# OUTPUT
A JSON dict with the following structure:
{
    description: "..."
}
"""

conditioned_frame = """Graph:
Target trait:
$passage
"""

one_shot_paragraph = """
{'att|obj': [('crowded', 'movie theater'),
  ('sitting in middle', 'Ye'),
  ('wrong film', 'film'),
  ('has started', 'film')],
 'obj-obj': [('Ye', 'inside', 'movie theater'), ('Ye', 'watching', 'film')],
 'att|obj-obj': [('sitting in middle', 'Ye', 'inside', 'movie theater'),
  ('sitting in middle', 'Ye', 'watching', 'film')],
 'obj-att|obj': [('Ye', 'inside', 'crowded', 'movie theater'),
  ('Ye', 'watching', 'wrong film', 'film'),
  ('Ye', 'watching', 'has started', 'film')],
 'att|obj-att|obj': [('sitting in middle',
   'Ye',
   'inside',
   'crowded',
   'movie theater'),
  ('sitting in middle', 'Ye', 'watching', 'wrong film', 'film'),
  ('sitting in middle', 'Ye', 'watching', 'has started', 'film')]}"""

one_shot_output = """
{
    "description": "'Ye is sitting in the middle of a crowded movie theater. Shortly after the film has started, Ye realize that Ye made a mistake in the cinema and ended up in the wrong film'"
}
"""

one_shot_trait = 'N'
one_shot_character = 'Ye'
prompt_template = [
    {'role': 'system', 'content': condition_system},
    {'role': 'user', 'content': Template(conditioned_frame).substitute(passage=one_shot_paragraph)},
    {'role': 'assistant', 'content': one_shot_output},
    {'role': 'user', 'content': conditioned_frame},
]
