from __future__ import annotations

from string import Template

vng_condition_system = """
# VNG Polisher Protocol

## 1. Objective

This protocol defines a rigorous, systematic approach for evaluating and refining Visual Narrative Grammar (VNG) prompts to ensure that:

- Each narrative unit (E, I, Pr, P) fulfills its intended function without revealing subsequent plot developments.
- Redundant or conflicting descriptive elements (e.g., facial expressions, gestures, object features, scene details) are accurately identified and resolved.
- The entire narrative process maintains logical coherence and aligns with the overarching VNG structure.
- Each VNG unit must independently describe a complete visual scene. No unit may rely on references to events or details that do not explicitly appear within itself (e.g., “focuses on part X, but X is not shown here”). When combined, all units should compose a multi-scene visual narrative.
- Prompts that already meet all standards remain unchanged.

## 2. Background

Visual Narrative Grammar (VNG) decomposes a visual scenario into distinct narrative units:

- **Establisher (E):** Introduces characters and setting to establish the context.
- **Initial (I):** Initiates action or conflict, revealing early plot tension.
- **Prolongation (Pr):** Acts as a transitional delay between Initial and Peak, allowing for pacing and buildup.
- **Peak (P):** Represents the climactic or turning point of the narrative.

> **Note:** Not all stories require all four units. Valid configurations include **E–I–Pr–P**, **E–I–P**, or **E–P** depending on narrative complexity.

## 3. Workflow

1. **Decomposition & Mapping:** Parse the input `situation` and assign narrative segments to corresponding VNG units, preserving the original sequence.
2. **Compatibility Review:**
   - Check for premature disclosure of later plot points (e.g., Peak elements appearing in E/I).
   - Identify repeated narrative cues (e.g., expressions, actions, lighting, color palette, composition) across units.
3. **Content Reallocation:**
   - Remove or reassign inappropriate elements to ensure narrative alignment.
   - Refine expressive details (e.g., facial expression, posture, lighting, palette, framing) to reinforce each unit’s unique narrative function.
4. **Continuity Assurance:**
   - Ensure consistency in character identity, attire, and environment across units.
   - Confirm a tide-like escalation of emotional or thematic intensity.
   - Ensure narrative functional differentiation between VNG units.
5. **Synthesis & Validation:**
   - Reassemble and output the polished VNG prompt set, ensuring strict adherence to narrative configuration.

## 4. Required Skills

- Familiarity with Visual Narrative Grammar and narrative theory
- Proficiency in prompt engineering for generative models
- Competence in evaluating narrative pacing and restructuring content
- Precision in maintaining semantic clarity and stylistic consistency

## 5. Input / Output

- **Input:**
  - `situation` (string): A detailed description of the scene
  - `VNG` (dict<string, string>): A mapping of units "E", "I", "Pr", "P" to their raw prompt descriptions

- **Output:**
  - A JSON object with a single property `VNG`, whose value is an object containing the refined prompt descriptions for each unit

**Example:**

```json
{
  "VNG": {
    "E": "Refined Establisher description",
    "I": "Refined Initial description",
    "Pr": "Refined Prolongation description",
    "P": "Refined Peak description"
  }
}
```

## 6. Constraints

- The output must preserve the input unit types and order; no units may be added, removed, or reordered.
- Only internal content of each unit may be modified.
- If the original descriptions already meet all criteria, they must be returned verbatim.

## 7. Related Concepts

- **Visual Narrative Grammar (VNG):** A theoretical framework for decomposing visual narratives
- **Prompt Engineering:** Techniques for crafting effective generative model inputs
- **Narrative Pacing:** Strategies for modulating the flow and intensity of story development
"""



narrative = 'Ye is sitting in the middle of a crowded movie theater. Shortly after the film has started, Ye realize that Ye made a mistake in the cinema and ended up in the wrong film'

vng_graph = """
{'E': "Create a 1024x1024 realistic image of Ye sitting in the middle of a crowded movie theater, scene's description: tight close-up shot capturing the back of the head and rows of silhouetted heads in front, unbalanced composition with numerous heads cutting horizontally through the frame, low-key lighting with flickering screen light creating uneasy shadows, monochromatic palette with washed-out whites and grays from the screen glow, a claustrophobic and tense atmosphere with dim ambient noise and rustling sounds, soft focus punctuated by occasional screen glare reflecting off heads, closed frame with boxed-in feeling due to the surrounding audience.",
 'I': "Create a 1024x1024 realistic image of Ye seated in a crowded movie theater, Ye's facial expression: Ye's eyes are widened in alarm, eyebrows raised and stretched straight, mouth open with lips drawn tight, nostrils slightly flared, Ye's gesture: Ye's body is tense, shoulders raised high, arms folded tightly over the chest in an attempt to make himself smaller, feet shifting nervously, with a slight lean forward as if ready to flee, scene's description: close-up with slightly canted angle to increase unease, frame within a frame using the gaps between seats to constrict view, low-key lighting with deep, harsh shadows cast by dim projector light, muted desaturated colors dominated by dark reds and grays, tense/suspenseful atmosphere with visible dappled patterns of light scattering unevenly, shallow focus to emphasize blurred and indistinct forms in the crowded setting, tight framing accentuating the claustrophobic feel within the row of densely packed seats, film's description: the film playing is a grainy/gritty depiction with a flickering, shaky projection creating an unsettling visual experience, employing dark, muted colors to evoke feelings of anxiety and discomfort, Ye's action: realized mistake in the film playing.",
 'P': "Create a 1024x1024 realistic image of Ye seated in a crowded movie theater watching a film, Ye's facial expression: Ye's eyes are wide open with slightly raised eyebrows, conveying alertness and confusion. His mouth is parted in a tense line, and his facial muscles are tight, creating a look of uncertainty and apprehension, Ye's gesture: Ye sits stiffly in his seat, shoulders lifted and slightly hunched as if trying to make himself smaller in the crowd. His hands rest tensely on the armrests or clench in his lap, and his feet shift nervously, scene's description: Close-up shot slightly from above, off-center framing with negative space emphasizing surrounding strangers, low-key lighting with a dim ambient glow from the screen reflecting unevenly, monochromatic palette of deep blacks and muted blues with occasional bright flashes from the screen, claustrophobic and oppressive atmosphere with shadows swallowing the edges of the frame, shallow focus on Ye's seat and immediate surrounding area, blurring distant elements, tight framing to create a sense of confinement within the theater seat, film's description: the film on screen is wrong—a misplayed reel showing unintended content, Ye's action: realizes he is watching the wrong film."}
"""

output = """
{
  "VNG": {
    "E": "Create a 1024x1024 realistic image of Ye sitting in the middle of a crowded movie theater, scene's description: tight close‑up shot capturing the back of the head and rows of silhouetted heads in front, unbalanced composition with numerous heads cutting horizontally through the frame, low‑key lighting with flickering screen light creating uneasy shadows, monochromatic palette with washed‑out whites and grays from the screen glow, a claustrophobic and tense atmosphere with dim ambient noise and rustling sounds, soft focus punctuated by occasional screen glare reflecting off heads, closed frame with boxed‑in feeling due to the surrounding audience.",
    "I": "Create a 1024x1024 realistic image of Ye seated in a crowded movie theater just as the film begins: Ye's facial expression: eyes fixed on the screen with focused anticipation, brow slightly furrowed, mouth relaxed; Ye's gesture: leaning forward subtly, hands gently resting on the armrests, feet planted firmly; scene's description: close‑up with a subtle Dutch angle to heighten expectancy, warm glow of opening titles casting soft highlights, gentle hum of the projector and quiet rustles filling the space, shallow depth of field isolating Ye against a blurred sea of silhouettes, muted blue‑gray palette punctuated by occasional warm tones, framing that emphasizes immersion in the unfolding film.",
    "P": "Create a 1024x1024 realistic image of Ye seated in a crowded movie theater watching a film, Ye's facial expression: Ye's eyes are wide open with slightly raised eyebrows, conveying alertness and confusion. His mouth is parted in a tense line, and his facial muscles are tight, creating a look of uncertainty and apprehension; Ye's gesture: Ye sits stiffly in his seat, shoulders lifted and slightly hunched as if trying to make himself smaller in the crowd. His hands rest tensely on the armrests or clench in his lap, and his feet shift nervously; scene's description: close‑up shot slightly from above, off‑center framing with negative space emphasizing surrounding strangers, low‑key lighting with a dim ambient glow from the screen reflecting unevenly, monochromatic palette of deep blacks and muted blues with occasional bright flashes from the screen, claustrophobic and oppressive atmosphere with shadows swallowing the edges of the frame, shallow focus on Ye's seat and immediate surrounding area with distant elements blurred, tight framing to create a sense of confinement within the theater seat; film's description: the film on screen is wrong—a misplayed reel showing unintended content; Ye's action: realizes he is watching the wrong film."
  }
}
"""


condition = """systematically reviews and refines each VNG unit to eliminate premature plot reveals and cross-unit redundancies while preserving the original structure:

SITUATION:
$passage

VNG:
$vng
"""
prompt_template = [
    {'role': 'system', 'content': vng_condition_system},
    {'role': 'user', 'content': Template(condition).substitute(passage=narrative, vng=vng_graph)},
    {'role': 'assistant', 'content': output},
    {'role': 'user', 'content': 'good, keep it up!'},
    {'role': 'assistant', 'content': 'ok, I will follow our previous conversation.'},
    {'role': 'user', 'content': condition},
]
