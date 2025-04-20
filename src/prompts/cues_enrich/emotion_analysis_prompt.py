from __future__ import annotations

from string import Template

condition_system = """
# Emotion Analysis Assistant

## GOAL

Analyze a single situation knowledge graph to extract and identify the emotion of the specified **Analyze Character** that best activates the target Big Five personality trait for the **Activate Character**. If no emotion is required to activate the trait, return an empty list.

## BACKGROUND

- **Trait Activation Theory (TAT)**: Personality traits are expressed only when situational cues that activate them are present. The core task is to determine which emotion the Analyze Character should exhibit in the scenario to most strongly activate the target trait.
- **Big Five Dimensions and Typical Activating Emotions**:
  - **O (Openness)**: Curious, imaginative; emotions reflecting wonder or interest can activate openness.
  - **C (Conscientiousness)**: Organized, responsible; emotions reflecting determination or seriousness can activate conscientiousness.
  - **E (Extraversion)**: Sociable, energetic; emotions reflecting excitement or enthusiasm can activate extraversion.
  - **A (Agreeableness)**: Compassionate, cooperative; emotions reflecting empathy or warmth can activate agreeableness.
  - **N (Neuroticism)**: Emotionally unstable, anxious; emotions reflecting fear, stress, or discomfort can activate neuroticism.

## EMOTIONS

Allowed output values:
```
'happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'contempt', 'neutral'
```

## INPUT

- **Situation Text**: The full narrative description of the scenario.
- **Knowledge Graph**: A single structured graph containing all nodes and edges (list of tuples).
- **Target Trait**: The Big Five personality dimension to be activated.
- **Activate Character**: The character whose trait activation is the focus.
- **Analyze Character**: The character whose emotion is to be determined.

## WORKFLOW

1. **Parse Input**: Validate Situation Text, Knowledge Graph, Target Trait, Activate Character, and Analyze Character.
2. **Overall Emotion Evaluation**:
   1. Review all tuples and context in the Knowledge Graph.
   2. Determine which emotion of the Analyze Character in this scenario best activates the target trait for the Activate Character.
   3. If multiple emotions could activate, choose the one with the highest relevance.
   4. If no emotion is needed to activate the trait, return an empty list.
3. **Format Output**: Generate JSON based on the evaluation.

## CONSTRAINTS

- The task is not simply selecting an emotion present in the graph, but choosing the one that best activates the trait.
- The output emotion must be one of the allowed EMOTIONS.
- Character names must match the input exactly.
- If no emotion is required to activate, explicitly return an empty list (`[]`).
- The final output must be a structured, machine-readable JSON.

## OUTPUT FORMAT

- **If an emotion is identified**, return:
  ```json
  {
    "character": "<Analyze Character>",
    "emotion": "<emotion>"
  }
  ```
- **If no emotion is required**, return:
  ```json
  {
    "character": "<Analyze Character>",
    "emotion": []
  }
  ```

## EXAMPLE

**Input**:
- **Target Trait**: Neuroticism
- **Situation Text**: "Ye is on the tram with a friend. At one stop, an attractive woman boards. As she passes, Ye's friend whistles after her. The woman turns and looks at Ye."
- **Knowledge Graph**:
  ```json
{'nodes': [
    ['object_2', {'type': 'object_node', 'value': 'friend'}],
    ['object_4', {'type': 'object_node', 'value': 'woman'}]],
 'edges': [
     ['object_2','object_4',{'type': 'relation_edge', 'value': 'whistles after'}]]}
  ```
- **Activate Character**: Ye
- **Analyze Character**: friend

**Output**:
```json
{
  "character": "friend",
  "emotion": "contempt"
}
```
"""

# Define common scenario for all examples
common_scenario = "Ye is on the tram with a friend. At one stop, an attractive woman gets on. As she passes Ye, Ye's friend whistles after her. The woman turns irritated and looks at Ye."

# Use the same scenario for all examples
few_shot_narrative_1 = common_scenario
few_shot_narrative_2 = common_scenario
few_shot_narrative_3 = common_scenario
few_shot_narrative_4 = common_scenario

# All examples use neuroticism as the target trait
few_shot_trait_1 = 'Neuroticism'
few_shot_trait_2 = 'Neuroticism'
few_shot_trait_3 = 'Neuroticism'
few_shot_trait_4 = 'Neuroticism'

few_shot_graph_1 = """
{'nodes': [['object_2', {'type': 'object_node', 'value': 'friend'}],
  ['object_4', {'type': 'object_node', 'value': 'woman'}]],
 'edges': [['object_2',
   'object_4',
   {'type': 'relation_edge', 'value': 'whistles after'}]]}
"""
few_shot_graph_2 = """
{'nodes': [['object_1', {'type': 'object_node', 'value': 'Ye'}],
  ['object_2', {'type': 'object_node', 'value': 'friend'}],
  ['object_3', {'type': 'object_node', 'value': 'tram'}]],
 'edges': [['object_1','object_2',{'type': 'relation_edge', 'value': 'with'}],
  ['object_1', 'object_3', {'type': 'relation_edge', 'value': 'on'}],
  ['object_2', 'object_3', {'type': 'relation_edge', 'value': 'on'}]]}
"""
few_shot_graph_3 = """
{'nodes': [
    ['object_1', {'type': 'object_node', 'value': 'Ye'}],
    ['object_4', {'type': 'object_node', 'value': 'woman'}],
    ['attribute|4|2', {'type': 'attribute_node', 'value': 'irritated'}]],
 'edges': [
     ['object_4','object_1',{'type': 'relation_edge', 'value': 'looks at'}],
     ['attribute|4|2', 'object_4', {'type': 'attribute_edge'}]]}
"""
few_shot_graph_4 = """
{'nodes': [
    ['object_1', {'type': 'object_node', 'value': 'Ye'}],
    ['object_4', {'type': 'object_node', 'value': 'woman'}],
    ['attribute|4|2', {'type': 'attribute_node', 'value': 'irritated'}]],
 'edges': [
     ['object_4','object_1',{'type': 'relation_edge', 'value': 'looks at'}],
     ['attribute|4|2', 'object_4', {'type': 'attribute_edge'}]]}
"""

few_shot_activate_character_1 = 'Ye'
few_shot_activate_character_2 = 'Ye'
few_shot_activate_character_3 = 'Ye'
few_shot_activate_character_4 = 'Ye'

few_shot_analyze_character_1 = 'friend'
few_shot_analyze_character_2 = 'friend'
few_shot_analyze_character_3 = 'woman'
few_shot_analyze_character_4 = 'Ye'


few_shot_output_1 = """
{
    "character": "friend",
    "emotion": "contempt"
}
"""
few_shot_output_2 = """
{
    "character": "friend",
    "emotion": "[]"
}
"""
few_shot_output_3 = """
{
    "character": "woman",
    "emotion": "anger"
}
"""
few_shot_output_4 = """
{
    "character": "Ye",
    "emotion": "fear"
}
"""


conditioned_frame = """Select the emotion of [analyze character] that best activate the target trait of [activate character] in the graph of the situation.:
Target trait:
$trait

Situation:
$passage

Knowledge Graph
$graph

Activate character:
$activate_character

Analyze character:
$analyze_character
"""

prompt_template = [
    {'role': 'system', 'content': condition_system},
    {'role': 'user', 'content': Template(conditioned_frame).substitute(passage=few_shot_narrative_1, trait=few_shot_trait_1,activate_character=few_shot_activate_character_1,analyze_character=few_shot_analyze_character_1, graph=few_shot_graph_1)},
    {'role': 'assistant', 'content': few_shot_output_1},
    {'role': 'user', 'content': Template(conditioned_frame).substitute(passage=few_shot_narrative_2, trait=few_shot_trait_2,activate_character=few_shot_activate_character_2,analyze_character=few_shot_analyze_character_2, graph=few_shot_graph_2)},
    {'role': 'assistant', 'content': few_shot_output_2},
    {'role': 'user', 'content': Template(conditioned_frame).substitute(passage=few_shot_narrative_3, trait=few_shot_trait_3,activate_character=few_shot_activate_character_3,analyze_character=few_shot_analyze_character_3, graph=few_shot_graph_3)},
    {'role': 'assistant', 'content': few_shot_output_3},
    {'role': 'user', 'content': Template(conditioned_frame).substitute(passage=few_shot_narrative_4, trait=few_shot_trait_4,activate_character=few_shot_activate_character_4,analyze_character=few_shot_analyze_character_4, graph=few_shot_graph_4)},
    {'role': 'assistant', 'content': few_shot_output_4},
    {'role': 'user', 'content': 'good, keep it up!'},
    {'role': 'assistant', 'content': 'ok, I will follow our previous conversation.'},
    {'role': 'user', 'content': conditioned_frame},
]
