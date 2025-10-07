from __future__ import annotations

from string import Template

vng_condition_system = """
# Structured storyboard maker

Your task is to construct a storyboard based on visual narrative grammar (VNG) from the given textual situation and its knowledge graph.
Return the result in JSON format of VNG, it has a key "VNG", and value is a dict where the keys are VNG elements and the values are the corresponding knowledge graph.

## Background

VNG includes four visual narrative elements (E, I, Pr, P), each serving a specific narrative function. These elements, when arranged together, form a coherent visual narrative arc.
Note: A narrative may not necessarily include all elements.
- Establisher (E): Sets the scene, introduces characters and environment.
- Initial (I): Introduces the main action or conflict, setting the stage for story development.
- Prolongation (Pr): A pacing element between Initial and Peak; can include dialogue, actions, or emotional transitions.
- Peak (P): Presents the climax or turning point of the narrative.

## Goals

- Deep understanding of the narrative Arc: Understand the narrative function of each element in VNG theory.
- Construct one knowledge graph for each VNG element: Grasp the story itself and write a knowledge graph for each element of the arc.
- Use E-I-P as the core narrative arc structure: Maintain a general transformation structure from scene setup (E), to conflict introduction (I), and then to climax (P).

## Exceptions

- If the narrative is overly simple, only a subset of arc elements may be used, e.g., E-P
- If the narrative is overly complex, multiple narrative arc elements may be used, e.g., E-I-P-Pr-P

## Constraints

- It is totally fine that some nodes and edges repeated in the different output knowledge graphs.
- Every panel's knowledge graph's nodes and edges should be CONSISTENT with the input knowledge graph.
- Every panel's knowledge graph's nodes should be connected to each other with edges from input knowledge graph.
- Every node and edge in the original knowledge graph should be included in the output knowledge graphs.
- The narrative arc should be a complete story; each visual narrative element must have a clear narrative function and logical connection.
- Unless the narrative is overly simple or complex, the arc should follow the E-I-P structure.
- DO NOT confuse the node's value with its identifier. For example, object_1's value is its corresponding content like "Ye", not "object_1".

## Workflows

1. Grasp the overall narrative and its knowledge graph.
2. Construct the VNG structure using the knowledge graph.
3. Check the coherence of narrative elements.
4. Make sure every panel's knowledge graph's nodes and relations are consistent with input knowledge graph.
"""



few_shot_narrative_1 = 'Ye is sitting in the middle of a crowded movie theater. Shortly after the film has started, Ye realize that Ye made a mistake in the cinema and ended up in the wrong film'

few_shot_graph_1 = """
{'nodes': [
  ['object_1', {'type': 'object_node', 'value': 'Ye'}],
  ['object_2', {'type': 'object_node', 'value': 'movie theater'}],
  ['object_3', {'type': 'object_node', 'value': 'film'}],
  ['attribute|2|1', {'type': 'attribute_node', 'value': 'crowded'}],
  ['attribute|3|1', {'type': 'attribute_node', 'value': 'wrong'}],
  ['attribute|3|2', {'type': 'attribute_node', 'value': 'has started'}]],
 'edges': [
  ['object_1','object_2',{'type': 'relation_edge', 'value': 'sitting in middle'}],
  ['object_1', 'object_3', {'type': 'relation_edge', 'value': 'watching'}],
  ['attribute|2|1', 'object_2', {'type': 'attribute_edge'}],
  ['attribute|3|1', 'object_3', {'type': 'attribute_edge'}],
  ['attribute|3|2', 'object_3', {'type': 'attribute_edge'}]]}
"""
few_shot_output_1 = """
{
  "VNG": {
    "E": {
      "nodes": [
        ["object_1", {"type": "object_node", "value": "Ye"}],
        ["object_2", {"type": "object_node", "value": "movie theater"}],
        ["attribute|2|1", {"type": "attribute_node", "value": "crowded"}]
      ],
      "edges": [
        ["object_1", "object_2", {"type": "relation_edge", "value": "sitting in middle"}],
        ["attribute|2|1", "object_2", {"type": "attribute_edge"}]
      ]
    },
    "I": {
      "nodes": [
        ["object_1", {"type": "object_node", "value": "Ye"}],
        ["object_3", {"type": "object_node", "value": "film"}],
        ["attribute|3|2", {"type": "attribute_node", "value": "has started"}]
      ],
      "edges": [
        ["object_1", "object_3", {"type": "relation_edge", "value": "watching"}],
        ["attribute|3|2", "object_3", {"type": "attribute_edge"}]
      ]
    },
    "P": {
      "nodes": [
        ["object_1", {"type": "object_node", "value": "Ye"}],
        ["object_3", {"type": "object_node", "value": "film"}],
        ["attribute|3|1", {"type": "attribute_node", "value": "wrong"}]
      ],
      "edges": [
        ["object_1", "object_3", {"type": "relation_edge", "value": "watching"}],
        ["attribute|3|1", "object_3", {"type": "attribute_edge"}]
      ]
    }
  }
}
"""

few_shot_narrative_2 = "Ye're on the tram with a friend. At one stop, an attractive woman gets on. As she passes Ye, Ye's friend whistles after her. The woman turns irritated and looks at Ye"

few_shot_graph_2 = """
{'nodes': [['object_1', {'type': 'object_node', 'value': 'Ye'}],
  ['object_2', {'type': 'object_node', 'value': 'friend'}],
  ['object_3', {'type': 'object_node', 'value': 'woman'}],
  ['object_4', {'type': 'object_node', 'value': 'tram'}],
  ['attribute|3|1', {'type': 'attribute_node', 'value': 'attractive'}],
  ['attribute|3|2', {'type': 'attribute_node', 'value': 'irritated'}]],
 'edges': [['object_1', 'object_4', {'type': 'relation_edge', 'value': 'on'}],
  ['object_2', 'object_4', {'type': 'relation_edge', 'value': 'on'}],
  ['object_2', 'object_3', {'type': 'relation_edge', 'value': 'whistles at'}],
  ['object_3', 'object_4', {'type': 'relation_edge', 'value': 'gets on'}],
  ['object_3', 'object_1', {'type': 'relation_edge', 'value': 'looks at'}],
  ['attribute|3|1', 'object_3', {'type': 'attribute_edge'}],
  ['attribute|3|2', 'object_3', {'type': 'attribute_edge'}]]}
"""

few_shot_output_2 = """
{
  "VNG": {
    "E": {
      "nodes": [
        ["object_1", {"type": "object_node", "value": "Ye"}],
        ["object_2", {"type": "object_node", "value": "friend"}],
        ["object_4", {"type": "object_node", "value": "tram"}]
      ],
      "edges": [
        ["object_1", "object_4", {"type": "relation_edge", "value": "on"}],
        ["object_2", "object_4", {"type": "relation_edge", "value": "on"}]
      ]
    },
    "I": {
      "nodes": [
        ["object_3", {"type": "object_node", "value": "woman"}],
        ["object_4", {"type": "object_node", "value": "tram"}],
        ["attribute|3|1", {"type": "attribute_node", "value": "attractive"}]
      ],
      "edges": [
        ["object_3", "object_4", {"type": "relation_edge", "value": "gets on"}],
        ["attribute|3|1", "object_3", {"type": "attribute_edge"}]
      ]
    },
    "Pr": {
      "nodes": [
        ["object_2", {"type": "object_node", "value": "friend"}],
        ["object_3", {"type": "object_node", "value": "woman"}]
      ],
      "edges": [
        ["object_2", "object_3", {"type": "relation_edge", "value": "whistles at"}]
      ]
    },
    "P": {
      "nodes": [
        ["object_1", {"type": "object_node", "value": "Ye"}],
        ["object_3", {"type": "object_node", "value": "woman"}],
        ["attribute|3|2", {"type": "attribute_node", "value": "irritated"}]
      ],
      "edges": [
        ["object_3", "object_1", {"type": "relation_edge", "value": "looks at"}],
        ["attribute|3|2", "object_3", {"type": "attribute_edge"}]
      ]
    }
  }
}

"""

few_shot_narrative_3 = "Ye give a presentation to the colleagues in Ye's department. As Ye speak, Ye notice that two of Ye's colleagues suddenly start laughing and whispering to each other"

few_shot_graph_3 = """
{'nodes': [['object_1', {'type': 'object_node', 'value': 'Ye'}],
  ['object_2', {'type': 'object_node', 'value': 'presentation'}],
  ['object_3', {'type': 'object_node', 'value': 'two colleagues'}],
  ['object_4', {'type': 'object_node', 'value': 'department'}],
  ['attribute|3|1', {'type': 'attribute_node', 'value': 'laughing'}],
  ['attribute|3|2',{'type': 'attribute_node', 'value': 'whispering to each other'}]],
 'edges': [['object_1', 'object_2', {'type': 'relation_edge', 'value': 'gives'}],
  ['object_1', 'object_4', {'type': 'relation_edge', 'value': 'belongs to'}],
  ['object_1', 'object_3', {'type': 'relation_edge', 'value': 'notices'}],
  ['object_2', 'object_4', {'type': 'relation_edge', 'value': 'in'}],
  ['object_3', 'object_4', {'type': 'relation_edge', 'value': 'belongs to'}],
  ['attribute|3|1', 'object_3', {'type': 'attribute_edge'}],
  ['attribute|3|2', 'object_3', {'type': 'attribute_edge'}]]}
"""

few_shot_output_3 = """
{
  "VNG": {
    "E": {
      "nodes": [
        ["object_1", {"type": "object_node", "value": "Ye"}],
        ["object_2", {"type": "object_node", "value": "presentation"}],
        ["object_4", {"type": "object_node", "value": "department"}]
      ],
      "edges": [
        ["object_1", "object_2", {"type": "relation_edge", "value": "gives"}],
        ["object_1", "object_4", {"type": "relation_edge", "value": "belongs to"}],
        ["object_2", "object_4", {"type": "relation_edge", "value": "in"}]
      ]
    },
    "P": {
      "nodes": [
        ["object_1", {"type": "object_node", "value": "Ye"}],
        ["object_2", {"type": "object_node", "value": "presentation"}],
        ["object_3", {"type": "object_node", "value": "two colleagues"}],
        ["object_4", {"type": "object_node", "value": "department"}],
        ["attribute|3|1", {"type": "attribute_node", "value": "laughing"}],
        ["attribute|3|2", {"type": "attribute_node", "value": "whispering to each other"}]
      ],
      "edges": [
        ["object_1", "object_2", {"type": "relation_edge", "value": "gives"}],
        ["object_1", "object_4", {"type": "relation_edge", "value": "belongs to"}],
        ["object_1", "object_3", {"type": "relation_edge", "value": "notices"}],
        ["object_2", "object_4", {"type": "relation_edge", "value": "in"}],
        ["object_3", "object_4", {"type": "relation_edge", "value": "belongs to"}],
        ["attribute|3|1", "object_3", {"type": "attribute_edge"}],
        ["attribute|3|2", "object_3", {"type": "attribute_edge"}]
      ]
    }
  }
}
"""

conditioned_frame = """Convert the following textual narrative and its knowledge graph into an storyboard constructed by VNG panels' knowledge graph:

Textual narrative:
$passage

Knowledge graph:
$graph
"""
prompt_template = [
    {'role': 'system', 'content': vng_condition_system},
    {
        'role': 'user', 'content': Template(
            conditioned_frame,
        ).substitute(passage=few_shot_narrative_1, graph=few_shot_graph_1),
    },
    {'role': 'assistant', 'content': few_shot_output_1},
    {
        'role': 'user', 'content': Template(
            conditioned_frame,
        ).substitute(passage=few_shot_narrative_2, graph=few_shot_graph_2),
    },
    {'role': 'assistant', 'content': few_shot_output_2},
    {
        'role': 'user', 'content': Template(
            conditioned_frame,
        ).substitute(passage=few_shot_narrative_3, graph=few_shot_graph_3),
    },
    {'role': 'assistant', 'content': few_shot_output_3},
    {'role': 'user', 'content': 'good, keep it up!'},
    {'role': 'assistant', 'content': 'ok, I will follow our previous conversation.'},
    {'role': 'user', 'content': conditioned_frame},
]
