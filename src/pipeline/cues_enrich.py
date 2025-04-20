from __future__ import annotations

from ..models.llms import TempletLLM
from .utils import find_key_in_result

emo_llm = TempletLLM('emotion_analysis')
exp_llm = TempletLLM('emotion_to_expression')
se_llm = TempletLLM('scene_enrich')
oe_llm = TempletLLM('object_enrich')

def make_expression(situation, trait, ana_character, act_character):
    # 1. Emotion Analysis
    res = emo_llm.call(
        passage=situation, trait=trait,
        analyze_character=ana_character,
        activate_character=act_character,
    )
    emotion = find_key_in_result(res, 'emotion')['emotion']
    # 2. Emotion to Expression
    res = exp_llm.call(passage=situation, emotion=emotion, character=ana_character)
    expression = find_key_in_result(res, 'expression')['expression']
    return expression

def make_scene(situation, character, trait, scene):
    """Generate the observable description of scene in situation to activate character's trait."""
    res = se_llm.call(
        passage=situation, character=character,
        trait=trait, scene=scene,
    )
    scene = find_key_in_result(res, 'scene')['scene']
    return scene

def make_object(situation, character, trait, object_):
    """Generate the observable description of object in situation to activate character's trait."""
    res = oe_llm.call(
        passage=situation, character=character,
        trait=trait, object=object_,
    )
    _object = find_key_in_result(res, 'object')['object']
    return _object

def enrich_characters(situation, trait, ana_characters, act_character):
    """"""''
    expressions = {
        ana_character: make_expression(situation, trait, ana_character, act_character)
        for ana_character in ana_characters
    }
    return expressions

def enrich_scenes(situation, trait, scenes, act_character):
    """"""''
    expressions = {
        scene: make_scene(situation, act_character, trait, scene)
        for scene in scenes
    }
    return expressions

def enrich_objects(situation, trait, objects, act_character):
    """"""''
    expressions = {
        object_: make_object(situation, act_character, trait, object_)
        for object_ in objects
    }
    return expressions
