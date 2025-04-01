from string import Template

vng_condition_system = """
# Text-to-Visual Narrative Script Converter

Your task is to construct an image sequence storyboard based on visual narrative grammar (VNG) from the given textual scenario.
Return the result in JSON format of VNG, it has a key "VNG", and key is a dict where the keys are VNG elements and the values are the corresponding panel descriptions.

## Background

VNG includes four visual narrative elements (E, I, Pr, P), each serving a specific narrative function. These elements, when arranged together, form a coherent visual narrative arc.
Note: A narrative may not necessarily include all elements.
- Establisher (E): Sets the scene, introduces characters and environment.
- Initial (I): Introduces the main action or conflict, setting the stage for story development.
- Prolongation (Pr): A pacing element between Initial and Peak; can include dialogue, actions, or emotional transitions.
- Peak (P): Presents the climax or turning point of the narrative.

## Goals

- Deep understanding of the narrative Arc: Understand the narrative function of each element in VNG theory.
- Construct a panel script for each VNG element: Grasp the story itself and write a panel script for each element of the arc.
- Use E-I-P as the core narrative arc structure: Maintain a general transformation structure from scene setup (E), to conflict introduction (I), and then to climax (P), ensuring fluency and coherence in storytelling.

## Exceptions

- If the narrative is overly simple, only a subset of arc elements may be used, e.g., E-P
- If the narrative is overly complex, multiple narrative arc elements may be used, e.g., E-I-P-Pr-P
- The number and type of narrative elements can be adjusted based on the context, but coherence and fluency must be maintained.

## Constraints
- DO NOT add visual narrative that are not derived from the textual narrative.
- The narrative arc should be a complete story; each visual narrative element must have a clear narrative function and logical connection.
- If different panels share the same characters or scenes, the expression used should remain consistent.
- If similar concepts in the text refer to the same entity, choose one expression and use it throughout. For example, “cinema” and “movie theater” both refer to the same entity — express both as “cinema.”
- Unless the narrative is overly simple or complex, the arc should follow the E-I-P structure.
- every panel should contain the full description of the narrative, including scene, characters, events and so on.

## Workflows
1. Grasp the overall narrative
2. Construct the panel scripts
3. Check the coherence of narrative elements
4. make sure all the entities are consistent, for example, if the text contains "cinema" and "movie theater", both refer to the same entity, express both as "cinema".
"""

vng_conditioned_re_frame = """Convert the following textual narrative into an image sequence storyboard:

Textual narrative:
$passage
"""

# Few-shot 示例数据
few_shot_narrative_1 = """You're on the tram with a friend. At one stop, an attractive woman gets on. 
As she passes you, your friend whistles after her.
The woman turns irritated and looks at you"""

few_shot_output_1 = """{"VNG": {
    "E": "on a tram, you and your friend sitting together in the seat",
    "I": "on a tram, at one stop, an attractive woman gets on standing in front of you and your friend",
    "Pr": "on a tram, you and your friend sitting together in the seat, your friend whistles after the attractive woman",
    "P": "on a tram, you and your friend sitting together in the seat, the attractive woman turns irritated and looks at you"
    }
}"""

few_shot_narrative_2 = """You are sitting in the middle of a crowded movie theater. 
Shortly after the film has started, you realize that you made a mistake in the cinema and ended up in the wrong film"""

few_shot_output_2 = """{"VNG": {
    "E": "In a crowed cinema, you are sitting in the middle of the cinema",
    "I": "In a crowed cinema, the film starts",
    "P": "In a crowed cinema, you realize that you made a mistake in the cinema and ended up in the wrong film"
    }
}"""

few_shot_narrative_3 = """You are invited to a friend's wedding and are chosen to take part in a pantomime game"""

few_shot_output_3 = """{"VNG": {
    "E": "In a wedding place, you are invited to a friend's wedding",
    "P": "In a wedding place, you are chosen to take part in a pantomime game"
    }
}"""

prompt_template = [
    {"role": "system", "content": vng_condition_system},
    {"role": "user", "content": Template(vng_conditioned_re_frame).substitute(passage=few_shot_narrative_1)},
    {"role": "assistant", "content": few_shot_output_1},
    {"role": "user", "content": Template(vng_conditioned_re_frame).substitute(passage=few_shot_narrative_2)},
    {"role": "assistant", "content": few_shot_output_2},
    {"role": "user", "content": Template(vng_conditioned_re_frame).substitute(passage=few_shot_narrative_3)},
    {"role": "assistant", "content": few_shot_output_3},
    {"role": "user", "content": vng_conditioned_re_frame}
]
