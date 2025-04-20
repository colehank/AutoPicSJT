from __future__ import annotations

from string import Template

system = """
# Cues Recognition Assistant

## GOAL

Help extract and identify the core cues that activate a target personality trait (Big Five) from a given situation knowledge graphs, and output the most representative cue for each segment in a structured, machine-readable JSON format.

## BACKGROUND

- **Trait Activation Theory (TAT)**: Trait Activation Theory posits that personality traits are expressed only when situational cues that activate those traits are present. The core task is to identify which scenario elements most strongly activate the target trait.
- **Big Five Dimensions and Typical Activating Cues**:
  - **O (Openness)**: Curious, imaginative; activating cues typically include novel, complex, or creative events.
  - **C (Conscientiousness)**: Organized, responsible; activating cues typically include situations requiring planning, rule-following, or task completion.
  - **E (Extraversion)**: Sociable, talkative; activating cues typically include social interactions, group settings, or leadership opportunities.
  - **A (Agreeableness)**: Compassionate, cooperative; activating cues typically include interpersonal conflict resolution, helping others, or cooperative tasks.
  - **N (Neuroticism)**: Emotionally unstable, anxious; activating cues typically include threatening, evaluative, or uncertain situations.
- **Knowledge graphs Structure**:
  The scenario is divided into multiple phase segments (e.g., E, I, P), each represented by one of five tuple patterns:
  1. `att|obj`: attribute → object
  2. `obj-obj`: object → relation → object
  3. `att|obj-obj`: attribute → object → relation → object
  4. `obj-att|obj`: object → relation → attribute → object
  5. `att|obj-att|obj`: attribute → object → relation → attribute → object

## INPUT

- **Situation Text**: The original narrative description of the scenario.
- **Knowledge Graphs**: A mapping of segments (keys) to lists of tuples following the above patterns.
- **Target Trait**: The Big Five personality dimension to be activated and recognized.

## WORKFLOW

1. **Parse Input**: Read and validate the Situation Text, Knowledge graphs, and Target Trait.
2. **Segment-wise Analysis**: For each segment key (e.g., E, I, P):
   1. Retrieve all tuples in that segment.
   2. Evaluate each tuple’s activation relevance to the Target Trait.
   3. If multiple highly relevant tuples exist, retain only the one with the highest relevance.
3. **Aggregate Results**: Collect the selected cue for each segment by key.
4. **Format Output**: Generate a top-level `"cues"` key mapping each segment to its list of cue objects.

## CONSTRAINTS

- Your task is not simply selecting tuples related to the trait, but activating the trait.
- MAKE SURE ALL THE OUTPUT NODES AND EDGES ORIGINATE FROM THE INPUT KNOWLEDGE graphs.
- Analyze each segment independently; do not match across keys.
- Output exactly one most representative cue per segment; if a segment has no activating cue, explicitly output an empty list (`[]`).
- The output cue must use one of the original tuple patterns and preserve the order of elements.
- The final result must be a structured, machine-readable JSON.
- There should at least one segment have cue.

## OUTPUT

Return a JSON object of the following structure:
```json
{
  "cues": {
    "E": [ { "type": "att|obj", "content": ["exampleAttribute", "exampleObject"] } ],
    "I": [],
    "P": [ { "type": "obj-att|obj", "content": ["exampleObject", "exampleRelation", "exampleAttribute", "exampleObject"] } ]
  }
}
```
"""

one_shot_trait = 'Neuroticism'

one_shot_paragraph = """
You are sitting in the middle of a crowded movie theater.
Shortly after the film has started, you realize that you made a mistake in the cinema and ended up in the wrong film.
"""

one_shot_graph = """
{'E': {'att|obj': [('crowded', 'movie theater')],
  'obj-obj': [('Ye', 'sitting in', 'movie theater')],
  'att|obj-obj': [],
  'obj-att|obj': [('Ye', 'sitting in', 'crowded', 'movie theater')],
  'att|obj-att|obj': []},
 'I': {'att|obj': [],
  'obj-obj': [('film', 'started in', 'movie theater')],
  'att|obj-obj': [],
  'obj-att|obj': [],
  'att|obj-att|obj': []},
 'P': {'att|obj': [('wrong', 'film')],
  'obj-obj': [('Ye', 'ended up in', 'film')],
  'att|obj-obj': [],
  'obj-att|obj': [('Ye', 'ended up in', 'wrong', 'film')],
  'att|obj-att|obj': []}
}
"""

one_shot_output = """
{
  "cues": {
    "E": [ { "type": "att|obj", "content": ["crowded","movie theater"] } ],
    "I": [],
    "P": [ { "type": "obj-att|obj", "content": ["Ye","ended up in","wrong","film"] } ]
  }
}
"""

two_shot_trait = 'Agreeableness'

two_shot_paragraph = """
Ye argue with Ye's partner about a detail in a film Ye watched together.
Ye is absolutely sure that Ye's partner is wrong and Ye is right
"""

two_shot_graph = """
{'E': {'att|obj': [],
  'obj-obj': [('Ye', 'watched', 'film'), ('partner', 'watched', 'film')],
  'att|obj-obj': [],
  'obj-att|obj': [],
  'att|obj-att|obj': []},
 'I': {'att|obj': [],
  'obj-obj': [('Ye', 'argue with', 'partner'),
   ('Ye', 'about', 'detail'),
   ('partner', 'about', 'detail')],
  'att|obj-obj': [],
  'obj-att|obj': [],
  'att|obj-att|obj': []},
 'P': {'att|obj': [('absolutely sure', 'Ye'),
   ('believes right', 'Ye'),
   ('believes wrong', 'partner')],
  'obj-obj': [('Ye', 'argue with', 'partner')],
  'att|obj-obj': [('absolutely sure', 'Ye', 'argue with', 'partner'),
   ('believes right', 'Ye', 'argue with', 'partner')],
  'obj-att|obj': [('Ye', 'argue with', 'believes wrong', 'partner')],
  'att|obj-att|obj': [('absolutely sure',
    'Ye',
    'argue with',
    'believes wrong',
    'partner'),
   ('believes right', 'Ye', 'argue with', 'believes wrong', 'partner')]}}
"""

two_shot_output = """
{
  "cues": {
    "E": [],
    "I": [
      {
        "type": "obj-obj",
        "content": ["Ye", "argue with", "partner"]
      }
    ],
    "P": [
      {
        "type": "att|obj-att|obj",
        "content": ["believes right", "Ye", "argue with", "believes wrong", "partner"]
      }
    ]
  }
}

"""

four_shot_trait = 'Conscientiousnesss'

four_shot_paragraph = """
Ye watch TV in the evening and Ye is very tired.
Just as Ye is about to turn off the TV, Ye realize that one of Ye's favorite shows has just started.
Tomorrow Ye have a long working day ahead of Ye and have to get up early
"""

four_shot_graph = """
{'E': {'att|obj': [('in the evening', 'TV'), ('tired', 'Ye')],
  'obj-obj': [('Ye', 'watch', 'TV')],
  'att|obj-obj': [('tired', 'Ye', 'watch', 'TV')],
  'obj-att|obj': [('Ye', 'watch', 'in the evening', 'TV')],
  'att|obj-att|obj': [('tired', 'Ye', 'watch', 'in the evening', 'TV')]},
 'I': {'att|obj': [],
  'obj-obj': [('Ye', 'realize has started', 'favorite show')],
  'att|obj-obj': [],
  'obj-att|obj': [],
  'att|obj-att|obj': []},
 'P': {'att|obj': [('long', 'working day'),
   ('have to get up early', 'working day')],
  'obj-obj': [('Ye', 'have', 'working day')],
  'att|obj-obj': [],
  'obj-att|obj': [('Ye', 'have', 'long', 'working day'),
   ('Ye', 'have', 'have to get up early', 'working day')],
  'att|obj-att|obj': []
  }
}
"""

four_shot_output = """
{
  "cues": {
    "E": [],
    "I": [],
    "P": [
      {
        "type": "att|obj",
        "content": ["have to get up early", "working day"]
      }
    ]
  }
}

"""

conditioned_frame = """
SITUATION TEXT:
$passage

TARGET TRAIT:
$trait

KNOWLEDGE GRAPHS:
$graphs


MAKE SURE ALL THE OUT PUT NODES AND EDGES FROM THE INPUT KNOWLEDGE graphs AND RESPONSE IN JSON FORMAT!
"""

prompt_template = [
    {'role': 'system', 'content': system},
    {'role': 'user', 'content': Template(conditioned_frame).substitute(passage=one_shot_paragraph, graphs=one_shot_graph, trait=one_shot_trait)},
    {'role': 'assistant', 'content': one_shot_output},
    {'role': 'user', 'content': Template(conditioned_frame).substitute(passage=two_shot_paragraph, graphs=two_shot_graph, trait=two_shot_trait)},
    {'role': 'assistant', 'content': two_shot_output},
    {'role': 'user', 'content': Template(conditioned_frame).substitute(passage=four_shot_paragraph, graphs=four_shot_graph, trait=four_shot_trait)},
    {'role': 'assistant', 'content': four_shot_output},
    {'role': 'user', 'content': 'good, keep it up!'},
    {'role': 'assistant', 'content': 'ok, I will follow our previous conversation.'},
    {'role': 'user', 'content': conditioned_frame},
]
