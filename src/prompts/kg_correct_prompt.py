from __future__ import annotations

from string import Template

system = """Your task is task is to refine a knowledge graph by validating and aligning its tuples with the exact details of the original text.
You should:
1. Carefully analyze the original text to extract all key entities, attributes, and relationships.
2. Compare each triple in the knowledge graph with the textual descriptions to verify consistency.
3. Remove any triples that contradict or do not accurately reflect the information provided in the text.
4. Ensure that the final set of triples faithfully and comprehensively represents the content and context of the original narrative.

Respond in JSON.
"""

one_shot_paragraph = """You're on the tram with a friend.
At one stop, an attractive woman gets on.
As she passes you, your friend whistles after her.
The woman turns irritated and looks at you"""

one_shot_kg = """
{'att|obj': [('attractive', 'woman'), ('irritated', 'woman')],
 'obj-obj': [('Ye', 'on', 'tram'),
  ('friend', 'on', 'tram'),
  ('friend', 'whistles at', 'woman'),
  ('woman', 'gets on', 'tram'),
  ('woman', 'looks at', 'Ye')],
 'att|obj-obj': [('attractive', 'woman', 'gets on', 'tram'),
  ('attractive', 'woman', 'looks at', 'Ye'),
  ('irritated', 'woman', 'gets on', 'tram'),
  ('irritated', 'woman', 'looks at', 'Ye')],
 'obj-att|obj': [('friend', 'whistles at', 'woman', 'attractive'),
  ('friend', 'whistles at', 'woman', 'irritated')],
 'att|obj-att|obj': []}
"""

one_shot_output = """
{
 'att|obj': [
    ('attractive', 'woman'),
    ('irritated', 'woman')
 ],
 'obj-obj': [
    ('Ye', 'on', 'tram'),
    ('friend', 'on', 'tram'),
    ('friend', 'whistles at', 'woman'),
    ('woman', 'gets on', 'tram'),
    ('woman', 'looks at', 'Ye')
 ],
 'att|obj-obj': [
    ('attractive', 'woman', 'gets on', 'tram'),
    ('irritated', 'woman', 'looks at', 'Ye')
 ],
 'obj-att|obj': [],
 'att|obj-att|obj': []
}
"""

conditioned_frame = """
ORIGINAL_TEXT:
$passage

KNOWLEDGE_GRAPH:
$graph
"""

prompt_template = [
    {'role': 'system', 'content': system},
    {'role': 'user', 'content': Template(conditioned_frame).substitute(passage=one_shot_paragraph, graph=one_shot_kg)},
    {'role': 'assistant', 'content': one_shot_output},
    {'role': 'user', 'content': conditioned_frame},
]
