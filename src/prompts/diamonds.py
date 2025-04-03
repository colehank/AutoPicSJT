from string import Template
from ..datasets import DataManager

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
D_1: The phrase suggests that I must complete a task.
D_2: The phrase implies a task-oriented mindset.
D_3: The phrase indicates that I need to fulfill responsibilities.
I_1: The phrase conveys intellectual stimulation.
I_2: The phrase suggests an opportunity to showcase intellectual ability.
I_3: The phrase implies the need for deep thought.
A_1: The phrase suggests that I might be criticized.
A_2: The phrase implies that I may be blamed for something.
A_3: The phrase indicates the presence of a potential threat.
M1: The phrase hints at the presence of a potential sexual or romantic partner.
M2: The phrase emphasizes the importance of physical attractiveness.
M3: The phrase is full of sexual innuendos.
O1: The phrase makes the overall situation seem pleasant.
O2: The phrase conveys a playful and relaxed atmosphere.
O3: The phrase evokes feelings of joy and relaxation.
N1: The phrase may trigger a sense of stress.
N2: The phrase conveys a sense of tension or anxiety.
N3: The phrase makes me feel frustrated.
Dc1: The phrase suggests that there is a possibility of deception.
Dc2: The phrase conveys the potential for dishonest behavior.
Dc3: The phrase implies that untruthfulness might be present.
S1: The phrase hints at the development of close interpersonal relationships.
S2: The phrase conveys signals of frequent communication.
S3: The phrase makes interacting with others appear welcoming.

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