# Auto Pictorial Situation Judgement Tests Generation

Auto PicSJT is a tool for automatically generating image-based Situation Judgement Tests (SJTs), specifically developed for BIGFIVE personality trait assessment.

## Project Overview

This project aims to use Large Language Models (LLMs) and image generation technology to automate the creation of visual situation judgment tests for evaluating individuals' performance on the five major personality traits:

- O: Openness
- C: Conscientiousness
- E: Extraversion
- A: Agreeableness
- N: Neuroticism

The system converts text-described situations into structured event graphs, extracts trait-related cues, and generates corresponding visual content, providing support for psychometric research and practice.

## Installation Guide

### Requirements

- Python 3.10+
- Various dependencies (see requirements.txt in the project root directory)

### Installation Steps

```bash
git clone https://github.com/colehank/AutoPicSJT.git
cd AutoPicSJT
pip install -r requirements.txt
```

## Usage

The project provides two main usage modes: single situation processing and batch processing of multiple situations.

### Single Situation Processing (EXAMPLE_single.py)

Here is an example of using single situation processing:

```python
from src.pipeline import SituationProcessor
import src

# Set result save path
results_dir = 'results/final'
trait = 'N'  # Trait identifier (O, C, E, A, N)
itemID = '1'  # Situation ID
you = 'Ye'    # Reference character
trait_ = 'Neuroticism'  # Full trait name

# Create result directories
this_dir = f'{results_dir}/{trait}'
fig_dir = f'{this_dir}/figs'
data_dir = f'{this_dir}/data'
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Load data
dm = src.DataManager()
situs = dm.read('situation_judgment_test', 'SJTs', extract_stiu=True)
situation = situs[trait][itemID]

# Process situation
P = SituationProcessor(model='gpt-4o', situ=situation, ref=you, trait=trait_)
res = P.fit(verbose=True)

# Visualize results
fig_G = src.draw_G(P.G)
fig_Gs = src.draw_Gs(P.Gs)
fig_intergarted_Gs = src.draw_Gs(P.intergerated_Gs)

# Save results
fig_G.savefig(f'{fig_dir}/G_{trait}_{itemID}.tif', dpi=300, bbox_inches='tight')
fig_Gs.savefig(f'{fig_dir}/Gs_{trait}_{itemID}.tif', dpi=300, bbox_inches='tight')
fig_intergarted_Gs.savefig(f'{fig_dir}/GsEnriched_{trait}_{itemID}.tif', dpi=300, bbox_inches='tight')
```

### Batch Processing (EXAMPLE_batch.py)

Example of batch processing multiple situations:

```python
from src.pipeline import SituationProcessor
import src
import concurrent.futures
import json
import pickle

# Set parameters
results_dir = 'results/final'
traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
n_item = 21  # Number of situations per trait
max_workers = len(traits) * n_item  # Maximum parallel working threads

# Create task list
tasks = [(trait, str(item_id)) for trait in traits for item_id in range(n_item + 1)]

# Load data
dm = src.DataManager()
situs = dm.read('situation_judgment_test', 'SJTs', extract_stiu=True)

# Define processing task
def task(trait, item_id, model='gpt-4o', ref_you='Ye', max_attempts=10):
    situ = situs[trait[0]][item_id]
    attempts = 0
    while attempts < max_attempts:
        try:
            P = src.SituationProcessor(
                model=model,
                situ=situ,
                trait=trait,
                ref=ref_you,
            )
            res = P.fit(verbose=False)
            return trait, item_id, res
        except Exception as e:
            attempts += 1
            if attempts >= max_attempts:
                return trait, item_id, None

# Process all tasks in parallel
all_results = {trait: {} for trait in traits}
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(task, trait, item_id): (trait, item_id) for trait, item_id in tasks}
    for future in concurrent.futures.as_completed(futures):
        trait, item_id, res = future.result()
        all_results[trait][item_id] = res

# Save results
for trait, items in all_results.items():
    this_dir = f'{results_dir}/{trait[0]}'
    fig_dir = f'{this_dir}/figs'
    data_dir = f'{this_dir}/data'
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    for item_id, res in items.items():
        if res is None:
            continue

        # Save images
        fig_G = src.draw_G(res['situation_graph'])
        fig_Gs = src.draw_Gs(res['vng_graphs'])
        fig_intergarted_Gs = src.draw_Gs(res['intergrated_Gs'])
        fig_G.savefig(f'{fig_dir}/G_{trait[0]}_{item_id}.tif', dpi=300, bbox_inches='tight')
        fig_Gs.savefig(f'{fig_dir}/Gs_{trait[0]}_{item_id}.tif', dpi=300, bbox_inches='tight')
        fig_intergarted_Gs.savefig(f'{fig_dir}/GsEnriched_{trait[0]}_{item_id}.tif', dpi=300, bbox_inches='tight')

        # Save data
        with open(f'{data_dir}/{trait[0]}_{item_id}_all.pkl', 'wb') as f:
            pickle.dump(res, f)

        # Save results in json format
        for re in res:
            if re in ['cues', 'enriched_cues', 'Gs_prompt']:
                with open(f'{data_dir}/{trait[0]}_{item_id}_{re}.json', 'w') as f:
                    json.dump(res[re], f, indent=4)
```

## Project Structure

The main components of the project include:

- `src/`: Core source code directory
  - `datasets/`: Dataset management
  - `models/`: Language model interfaces
  - `pipeline/`: Situation processing pipeline
  - `prompts/`: Prompt templates
  - `utils/`: Utility functions
  - `viz/`: Visualization tools
- `psychometrics/`: Psychometric tools
- `sta/`: Statistical analysis tools
- `results/`: Result storage directory
  - `event_graph/`: Event graphs
  - `final/`: Final results
  - `situ_character/`: Situation and character data
  - `trait_cues/`: Trait cues
  - `vng_graph/`: Visual narrative graphs

## Main Process

1. **Situation Parsing**: Convert text situations into event graphs
2. **Cue Extraction**: Extract cues related to the target trait from the graph
3. **Cue Enrichment**: Enrich and expand the extracted cues
4. **Visual Narrative Generation**: Generate visual narrative graphs
5. **Result Integration and Visualization**: Integrate results and generate visual outputs

## Parameter Description

- `model`: Language model used (default: 'gpt-4o')
- `situ`: Situation text description
- `ref`: Reference character name
- `trait`: Target trait name
- `verbose`: Whether to display detailed processing

## Output Description

The processing pipeline generates multiple results:

1. **Event Graph (G)**: Structured representation of the situation
2. **Visual Narrative Graph (Gs)**: Visual narrative structure based on the situation
3. **Integrated Graph (intergerated_Gs)**: Enriched visual narrative graph
4. **Cue Data**: Extracted trait-related cues
5. **Visualizations**: Visual representations of the above graphs

## License

Please refer to the [LICENSE](LICENSE) file.
