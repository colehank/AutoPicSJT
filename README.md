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

Here is an example of using single situation processing, input is the situation from a SJT's item:

```python
import src
P = SituationProcessor(model='gpt-4o', situ=situation, trait=trait)
res = P.fit(verbose=True)
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
