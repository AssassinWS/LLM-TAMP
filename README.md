# LLM-TAMP

Large Language Model-based Task and Motion Planning with Motion Failure Reasoning

## Perequisite

```bash
pip install -r requirements.txt
```

## Usage

We use `hydra-core` to configure the project.

## Folder Description
- `assets`: robots configurations
- `configs`: parameters for the environment and planners
- `envs`: the developed environment based on Pybullet
- `planners`: connecting LLM with TAMP
- `prompts`: prompt templates
- `utils`: miscellanies

## Before Running

fill in the `openai_keys`/openai_key.json. with your own openAI API key

### Run TAMP planning

Full example with various options:

```bash
python main.py --config-name=llm_tamp env=easy_box_small_basket planner=llm_backtrack max_llm_calls=10 overwrite_instances=true play_traj=true use_gui=true
```

- `env`: the environment setting, see `configs/env`
- `planner`: the planner, see `configs/planner`
- `max_llm_calls`: max number of llm calls
- `overwrite_instances`: we create & load task instances (with different init state) saved in `envs/task_instances`. Set `overwrite_instances` to true to recreate & save task instances
- `play_traj`: whether to play motion trajectory in pybullet
- `use_gui`: whether enable gui in pybullet

### Run parameter sampling

Run with LLM sampler:

```bash
python main.py --config-name=llm_tamp env=easy_box_small_basket planner=llm_sample_params max_llm_calls=10 play_traj=true use_gui=true
```

Run with random sampler:

```bash
python main.py --config-name=random_sample env=easy_box_small_basket
```