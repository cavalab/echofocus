# EchoFocus

EchoFocus is an AI method for echocardiography that diagnoses and measures cardiac function based on videos comprising an echocardiogram study.   
Its name refers to the fact that EchoFocus skips view classification, instead relying on attention mechanisms to determine which echo views to priortize in making specfic predictions. 

# Install

Project dependencies are in `pyproject.toml` and `uv.lock`. 

## Requirements:

- Python >= 3.9
- Cuda-enabled NVIDA GPU 

## Quickstart (uv)

Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync # sync dependencies
uv run echofocus.py --help
```

## Quickstart (pip)

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python echofocus.py --help
```

# Usage

## Train Models

To train a model, specify a model name, a dataset, and task. 

```
python echofocus.py train --model_name [model_name] --dataset [dataset] --task [measure,chd,fyler]
```

- Trained models and training results are stored in `./trained_models/[model_name]`. If the `model_name` you specify already exists, that one will be loaded from the best checkpoint and training will resume according to the specified arguments.
- Task information is stored in `config.json`. 
- The `dataset` is used as a key lookup to `config['dataset'][dataset]` to load the paths to the label files.

## Generate Study Embeddings

```
python echofocus.py embed
```

## Explain Model Outputs

Model explanations can be generated using integrated gradients. 

```
python echofocus.py explain
```

# Contact

This work is a joint project of the [Congenital Heart AI Lab (CHAI Lab)](https://research.childrenshospital.org/research-units/congenital-heart-artificial-intelligence-lab) and the [Cava Lab](https://cavalab.org) at Boston Children's Hospital, affiliated with Harvard Medical School.

To get help with the repository, [create an issue](https://github.com/cavalab/echofocus/issues). 
PR contributions are very welcome. 

## Maintainers

- William G. La Cava ([@lacava](https://github.com/lacava))
- Platon Lukyanenko
- Joshua Mayourian

# Acknowledgments

The authors would like to acknowledge Boston Children's Hospital's High-Performance Computing Resources Clusters Enkefalos 3 (E3) made available for conducting the research reported in this publication.

This work was supported in part by the Kostin Innovation Fund, Thrasher Research Fund Early Career Award, NIH/NHLBI T32HL007572, and NIH/NLHBI 2U01HL098147-12.