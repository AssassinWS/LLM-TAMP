# LLM-TAMP

This is the official repository of the paper:

[LLM3: Large Language Model-based Task and Motion Planning with Motion Failure Reasoning](https://arxiv.org/abs/2403.11552). 

$\text{LLM}^3$ is an LLM-powered Task and Motion Planning (TAMP) framework that leverages a pre-trained LLM (GPT-4) as the task planner, parameter sampler, and motion failure reasoner. We evaluate the framework in a series of tabletop box-packing tasks in Pybullet.

# Demo





https://github.com/AssassinWS/LLM-TAMP/assets/144423427/74566b14-a62e-401d-a8d9-2f27f3a7ede3










# Prerequisite

## Install dependencies

```bash
git clone git@github.com:AssassinWS/LLM-TAMP.git
cd LLM-TAMP
pip install -r requirements.txt
```

## Project structure
- `assets`: robots configurations and environment assets
- `configs`: config parameters for the environment and planners
- `envs`: the developed environment based on Pybullet
- `task_instances`: randomly generated task instances
- `planners`: TAMP planners
- `prompts`: prompt templates
- `utils`: utility functions

We use `hydra-core` to configure the project.


# Usage

## Before Running

First, create a folder `openai_keys` under the project directory; Second, create a file `openai_key.json` under the folder `openai_keys`; Third, fill in this json file with your openAI API key:

```bash
{
    "key": "",
    "org": "",
    "proxy" : ""
}
```

## Run TAMP planning
The ablation study in the LLM^3 paper.

Full example with various options:

```bash
python main.py --config-name=llm_tamp env=easy_box_small_basket planner=llm_backtrack max_llm_calls=10 overwrite_instances=true play_traj=true use_gui=true
```

- `env`: the environment setting, see `configs/env`
- `planner`: the planner, see `configs/planner`
- `max_llm_calls`: max number of LLM calls
- `overwrite_instances`: we create & load task instances (with different init states) saved in `envs/task_instances`. set `overwrite_instances` to true to recreate & save task instances
- `play_traj`: whether to play motion trajectory in Pybullet
- `use_gui`: whether enable gui in Pybullet

## Run parameter sampling
The action parameter selection experiment in the LLM^3 paper.

Run with the LLM sampler:

```bash
python main.py --config-name=llm_tamp env=easy_box_small_basket planner=llm_sample_params max_llm_calls=10 play_traj=true use_gui=true
```

Run with the random sampler:

```bash
python main.py --config-name=random_sample env=easy_box_small_basket
```
