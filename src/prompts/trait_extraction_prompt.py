from __future__ import annotations

from string import Template

system = """
Extract tuples(cues) that best ACTIVATE the input specified Big Five personality trait (e.g., O(Openness), C(Conscientiousness), E(Extraversion), A(Agreeableness), N(Neuroticism)) from the given situation text and its associated knowledge graph.

## GOAL
Extract tuples(cues) that best ACTIVATE the input trait from the given situation text and its associated knowledge graph.
MAKE SURE ALL THE OUT PUT NODES AND EDGES FROM THE INPUT KNOWLEDGE GRAPH

## BACKGROUND
- Trait Activation Theory and Big Five Personality
Trait Activation Theory (TAT) suggests that whether a personality trait is expressed depends on whether the situation provides cues that activate the trait.
Below is a brief overview of how each of the Big Five traits is typically activated.
    - O(Openness):Curious, imaginative, open to new experiences; Activating Cues: Novel, creative, or complex events and so on.
    - C(Conscientiousnesss): Organized, responsible, and goal-oriented; Activating Cues: Situations requiring planning, rule-following, or task completion and so on.
    - E(Extraversion): Sociable, talkative, and energetic; Activating Cues: Social settings or contexts that involve interaction or leadership and so on.
    - A(Agreeableness): Compassionate, cooperative, and empathetic; Activating Cues: Interpersonal conflict, opportunities for empathy or helping and so on.
    - N(Neuroticism): Emotionally unstable, anxious, reactive to stress; Activating Cues: Threatening, evaluative, or uncertain situations and so on.

- Inputs:
1. A situation description text, which may imply certain emotions, behaviors or attitudes.
2. A knowledge graph represented as a dictionary of phrase-type keys (e.g. att|obj, obj-obj, att|obj-obj, etc.), each mapping to a list of tuples that describe attributes, relations or combined attribute-relation phrases.
3. A target Big Five trait (one of: Openness, Conscientiousnesss, Extraversion, Agreeableness, Neuroticism).
Some graph tuples may overlap in content; careful selection is required to pick those most representative of the target trait.

## WORKFLOW

1. Identify the Target Trait
Read the input specification to determine which of the Big Five traits is to be activated.

2. Analyze the situation Text
Understand the context, actions, and implied emotions or attitudes in the text.

3. Scan the Knowledge Graph
Look for tuples containing keywords or relations can actiavte the target trait.

4. Checkiing for Activation
Cues/tuples that activate the trait can make dilemma, people of different traits behave differently in the same cues.

5. Resolve Overlaps
When multiple tuples from different categories overlap, select the one that most clearly activate the target trait.

6. Ensure Diversity
Choose tuples that describe distinct aspects or events—no two should convey essentially the same information.

7. Ensure Integrity
Ensure that the extracted tuple combinations can jointly and completely describe the shortest event that activates the trait.

8. IMPORTENT
MAKE SURE ALL THE OUT PUT NODES AND EDGES FROM THE INPUT KNOWLEDGE GRAPH

## CONSTRAINTS

- Trait-Actiavtion: Every extracted tuple must clearly activate the specified Big Five trait.

- Non-Redundancy: Selected tuples must be mutually distinct in content.

- Extracted cues number: the extracted cues number MUST BE MORE TAHN 0, it can be more than 2 or less than 2.

- Format Compliance: Output must be a list of dictionaries, each with:

    "type": the phrase-type key from the knowledge graph.

    "content": the exact tuple from that category.

- Your task is not simple select the tuples relate to the trait, but actiavte trait.
- MAKE SURE ALL THE OUT PUT NODES AND EDGES FROM THE INPUT KNOWLEDGE GRAPH


## OUTPUT
A JSON objects of dicts.

"type": The type of the tuple from the knowledge graph (e.g., 'att|obj-obj', 'obj-obj').

"content": A tuple that provides the detailed description (e.g., ('irritated', 'woman', 'looks at', 'Ye')).

### Example Input

TARGET TRAIT:
Neuroticism

SITUATION TEXT:
You're on the tram with a friend. At one stop, an attractive woman gets on.
As she passes you, your friend whistles after her.
The woman turns irritated and looks at you

KNOWLEDGE GRAPH:
{'att|obj': [('attractive', 'woman'), ('irritated', 'woman')],
 'obj-obj': [('Ye', 'on', 'tram'),
  ('friend', 'on', 'tram'),
  ('friend', 'whistles after', 'woman'),
  ('woman', 'gets on', 'tram'),
  ('woman', 'looks at', 'Ye')],
 'att|obj-obj': [('attractive', 'woman', 'gets on', 'tram'),
  ('attractive', 'woman', 'looks at', 'Ye'),
  ('irritated', 'woman', 'gets on', 'tram'),
  ('irritated', 'woman', 'looks at', 'Ye')],
 'obj-att|obj': [('friend', 'whistles after', 'attractive', 'woman'),
  ('friend', 'whistles after', 'irritated', 'woman')],
 'att|obj-att|obj': []}

### Example Output:
{
    'cues':[
            {'type': 'att|obj-obj', 'content': ['irritated', 'woman', 'looks at', 'Ye']},
            {'type': 'obj-obj', 'content': ['friend', 'whistles after', 'woman']}
        ]
}

"""

one_shot_trait = 'Neuroticism'

one_shot_paragraph = """
You are sitting in the middle of a crowded movie theater.
Shortly after the film has started, you realize that you made a mistake in the cinema and ended up in the wrong film.
"""

one_shot_graph = """
{'att|obj': [('crowded', 'movie theater'),
  ('sitting in middle', 'Ye'),
  ('wrong film', 'film'),
  ('has started', 'film')],
 'obj-obj': [('Ye', 'inside', 'movie theater'), ('Ye', 'watching', 'film')],
 'att|obj-obj': [('sitting in middle', 'Ye', 'inside', 'movie theater'),
  ('sitting in middle', 'Ye', 'watching', 'film')],
 'obj-att|obj': [('Ye', 'inside', 'movie theater', 'crowded'),
  ('Ye', 'watching', 'film', 'wrong film'),
  ('Ye', 'watching', 'film', 'has started')],
 'att|obj-att|obj': [('sitting in middle',
   'Ye',
   'inside',
   'movie theater',
   'crowded'),
  ('sitting in middle', 'Ye', 'watching', 'film', 'wrong film'),
  ('sitting in middle', 'Ye', 'watching', 'film', 'has started')]}
"""

one_shot_output = """
{
    'cues':[
                {
                    'type': 'obj-att|obj',
                    'content': ['Ye', 'watching', 'wrong film', 'film']
                },
                {
                    'type': 'att|obj-att|obj',
                    'content': ['sitting in middle', 'Ye', 'inside', 'crowded', 'movie theater']
                }
            ]
}
"""

two_shot_trait = 'Agreeableness'

two_shot_paragraph = """
Ye argue with Ye's partner about a detail in a film Ye watched together.
Ye is absolutely sure that Ye's partner is wrong and Ye is right
"""

two_shot_graph = """
{'att|obj': [('absolutely sure of being right', 'Ye'),
  ('believed to be wrong', 'partner'),
  ('detail disputed', 'film')],
 'obj-obj': [('Ye', 'argues with', 'partner'),
  ('Ye', 'watched', 'film'),
  ('partner', 'watched', 'film')],
 'att|obj-obj': [('absolutely sure of being right',
   'Ye',
   'argues with',
   'partner'),
  ('absolutely sure of being right', 'Ye', 'watched', 'film'),
  ('believed to be wrong', 'partner', 'watched', 'film')],
 'obj-att|obj': [('Ye', 'argues with', 'believed to be wrong', 'partner'),
  ('Ye', 'watched', 'detail disputed', 'film'),
  ('partner', 'watched', 'detail disputed', 'film')],
 'att|obj-att|obj': [('absolutely sure of being right',
   'Ye',
   'argues with',
   'believed to be wrong',
   'partner'),
  ('absolutely sure of being right',
   'Ye',
   'watched',
   'detail disputed',
   'film'),
  ('believed to be wrong', 'partner', 'watched', 'detail disputed', 'film')]}
"""

two_shot_output = """
{
  "cues": [
    {
      "type": "att|obj-att|obj",
      "content": ["absolutely sure of being right", "Ye", "argues with", "believed to be wrong", "partner"]
    },
    {
      "type": "obj-obj",
      "content": ["Ye", "argues with", "partner"]
    }
  ]
}
"""

three_shot_trait = 'Extraversion'

three_shot_paragraph = """
Ye want to watch a match of Ye's national team at the World Cup.
The game in the stadium has been sold out for a long time.
But there is several other ways to watch the game
"""

three_shot_graph = """
{'att|obj': [('sold out', 'game'), ('has other viewing options', 'game')],
 'obj-obj': [('Ye', 'wants to watch', 'game'),
  ('national team', 'belongs to', 'Ye'),
  ('game', 'part of', 'World Cup'),
  ('game', 'played in', 'stadium')],
 'att|obj-obj': [('sold out', 'game', 'part of', 'World Cup'),
  ('sold out', 'game', 'played in', 'stadium'),
  ('has other viewing options', 'game', 'part of', 'World Cup'),
  ('has other viewing options', 'game', 'played in', 'stadium')],
 'obj-att|obj': [('Ye', 'wants to watch', 'sold out', 'game'),
  ('Ye', 'wants to watch', 'has other viewing options', 'game')],
 'att|obj-att|obj': []}
"""

three_shot_output = """
{
  "cues": [
    {
      "type": "att|obj",
      "content": ["sold out", "game"]
    },
    {
      "type": "att|obj",
      "content": ["has other viewing options", "game"]
    },
    {
      "type": "obj-obj",
      "content": ["Ye", "wants to watch", "game"]
    }
  ]
}
"""

four_shot_trait = 'Conscientiousnesss'

four_shot_paragraph = """
Ye watch TV in the evening and Ye is very tired.
Just as Ye is about to turn off the TV, Ye realize that one of Ye's favorite shows has just started.
Tomorrow Ye have a long working day ahead of Ye and have to get up early
"""

four_shot_graph = """
{'att|obj': [('very tired', 'Ye'),
  ('long', 'working day'),
  ('tomorrow', 'working day'),
  ('needs to get up early', 'Ye')],
 'obj-obj': [('Ye', 'about to turn off', 'TV'),
  ('Ye', 'has ahead', 'working day'),
  ('favorite show', 'started on', 'TV')],
 'att|obj-obj': [('very tired', 'Ye', 'about to turn off', 'TV'),
  ('very tired', 'Ye', 'has ahead', 'working day'),
  ('needs to get up early', 'Ye', 'about to turn off', 'TV'),
  ('needs to get up early', 'Ye', 'has ahead', 'working day')],
 'obj-att|obj': [('Ye', 'has ahead', 'long', 'working day'),
  ('Ye', 'has ahead', 'tomorrow', 'working day')],
 'att|obj-att|obj': [('very tired', 'Ye', 'has ahead', 'long', 'working day'),
  ('very tired', 'Ye', 'has ahead', 'tomorrow', 'working day'),
  ('needs to get up early', 'Ye', 'has ahead', 'long', 'working day'),
  ('needs to get up early', 'Ye', 'has ahead', 'tomorrow', 'working day')]}
"""

four_shot_output = """
{
  "cues": [
    {
      "type": "att|obj-att|obj",
      "content": ["needs to get up early", "Ye", "has ahead", "long", "working day"]
    },
    {
      "type": "obj-obj",
      "content": ["favorite show", "started on", "TV"]
    },
  ]
}
"""

conditioned_frame = """
TARGET TRAIT:
$trait

SITUATION TEXT:
$passage

KNOWLEDGE GRAPH:
$graph


MAKE SURE ALL THE OUT PUT NODES AND EDGES FROM THE INPUT KNOWLEDGE GRAPH AND RESPONSE IN JSON FORMAT!
"""

prompt_template = [
    {'role': 'system', 'content': system},
    {'role': 'user', 'content': Template(conditioned_frame).substitute(passage=one_shot_paragraph, graph=one_shot_graph, trait=one_shot_trait)},
    {'role': 'assistant', 'content': one_shot_output},
    {'role': 'user', 'content': Template(conditioned_frame).substitute(passage=two_shot_paragraph, graph=two_shot_graph, trait=two_shot_trait)},
    {'role': 'assistant', 'content': two_shot_output},
    {'role': 'user', 'content': Template(conditioned_frame).substitute(passage=three_shot_paragraph, graph=three_shot_graph, trait=three_shot_trait)},
    {'role': 'assistant', 'content': three_shot_output},
    {'role': 'user', 'content': Template(conditioned_frame).substitute(passage=four_shot_paragraph, graph=four_shot_graph, trait=four_shot_trait)},
    {'role': 'assistant', 'content': four_shot_output},
    {'role': 'user', 'content': 'good, keep it up!'},
    {'role': 'assistant', 'content': 'ok, I will follow our previous conversation.'},
    {'role': 'user', 'content': conditioned_frame},
]
