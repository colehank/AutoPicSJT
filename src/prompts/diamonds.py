# %%
from string import Template
from ..datasets import DataManager
# %%
diamonds = DataManager().read('situation_DIAMONDS', 'DIAMONDS')

_dim_system = """
You are a psychologist who is master at personality and situation.

## GOAL
Your task is to evaluate the extent to which a word or phrase in the given scenario aligns with the statements below, 
from 1 (not at all applicable) to 7 (very much applicable).

## Background
The DIAMONDS test consists of eight dimensions: 
D(Duty), I(Intellect), A(Adversity), M(Mating), O(pOsitivity), N(Negativity), Dc(Deception), S(Sociality).
Each dimension consists of 3 of statements that describe the situation.

## DIAMONDS test
D_1: I must complete a task.
D_2: I need to have a task-oriented mindset.
D_3: I must fulfill my (own) responsibilities.
I_1: This situation contains intellectual stimulation.
I_2: This situation provides the opportunity to showcase intellectual ability.
I_3: Information must be processed with deep thought.
A_1: I will be criticized.
A_2: I will be blamed for something.
A_3: I will be threatened by something or someone.
M_1: A potential sexual or romantic partner is present.
M_2: Physical attractiveness is important.
M_3: This situation is full of sexual innuendos.
O_1: This situation is pleasant.
O_2: This situation appears playful.
O_3: This situation is full of joy and relaxation.
N_1: This situation may trigger stress.
N_2: This situation may cause tension.
N_3: This situation may lead to a feeling of frustration.
Dc_1: It is possible to deceive others.
Dc_2: Someone in the situation may act dishonestly.
Dc_3: It is possible to be dishonest with others.
S_1: Close interpersonal relationships are important, or might develop.
S_2: Others will send many communication signals.
S_3: Communicating with others is important or welcome.

## OUTPUT
A JSON dict with the following structure:
{
    "DIAMONDS": {
                    "D_1": ...,
                    "D_2": ...,
                    "D_3": ...,
                    "I_1": ...,
                    "I_2": ...,
                    "I_3": ...,
                    "A_1": ...,
                    "A_2": ...,
                    "A_3": ...,
                    "M_1": ...,
                    "M_2": ...,
                    "M_3": ...,
                    "O_1": ...,
                    "O_2": ...,
                    "O_3": ...,
                    "N_1": ...,
                    "N_2": ...,
                    "N_3": ...,
                    "Dc_1": ...,
                    "Dc_2": ...,
                    "Dc_3": ...,
                    "S_1": ...,
                    "S_2": ...,
                    "S_3": ...
}
"""

conditioned_frame = """
SITUATION:
$passage

WORD/PHRASE:
$word
"""

one_shot_paragraph = """You are sitting in the middle of a crowded movie theater. 
Shortly after the film has started, you realize that you made a mistake in the cinema and ended up in the wrong film"""
one_shot_word = """crowded"""

one_shot_output = """{
    "DIAMONDS": [
        {
            "D_1": 1,
            "D_2": 1,
            "D_3": 1,
            "I_1": 1,
            "I_2": 1,
            "I_3": 1,
            "A_1": 6,
            "A_2": 6,
            "A_3": 6,
            "M_1": 1,
            "M_2": 1,
            "M_3": 1,
            "O_1": 1,
            "O_2": 1,
            "O_3": 1,
            "N_1": 6,
            "N_2": 6,
            "N_3": 6,
            "Dc_1": 1,
            "Dc_2": 1,
            "Dc_3": 1,
            "S_1": 1,
            "S_2": 1,
            "S_3": 1,
        }
    ]
}
"""

prompt_template = [
    {"role": "system", "content": Template(_dim_system).substitute(
        D = diamonds['Duty'],
        I = diamonds['Intellect'],
        A = diamonds['Adversity'],
        M = diamonds['Mating'],
        O = diamonds['pOsitivity'],
        N = diamonds['Negativity'],
        Dc = diamonds['Deception'],  # Changed D to DC for Deception
        S = diamonds['Sociality'],)},
    {"role": "user", "content": Template(conditioned_frame).substitute(
        passage=one_shot_paragraph,
        word=one_shot_word)},
    {"role": "assistant", "content": one_shot_output},
    {"role": "user", "content": conditioned_frame}
]