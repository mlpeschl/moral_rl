# Multi-Objective Reinforced Active Learning

## Dependencies
* wandb
* tqdm
* pytorch \>= 1.7.0
* numpy \>= 1.20.0
* scipy \>= 1.1.0
* pycolab == 1.2

## Weights and Biases
Our code depends on ![Weights and Biases](https://wandb.ai/) for visualizing and logging results during training.
As a result, we call `wandb.init()`, which will prompt to add an API key for linking the training runs with your 
personal wandb account. This can be done by pasting the `WANDB_API_KEY` into the respective box when running 
the code for the first time.

## Environments
Our gridworlds (Emergency: `randomized_v2.py`, Delivery: `randomized_v3.py`) build on the ![Pycolab](https://github.com/deepmind/pycolab) game engine with a custom wrapper
to provide similar functionality as the `gym` ![environments](https://github.com/openai/gym). This engine 
comes with a user interface and any `environment` can be played in the console using `python environment.py` 
with arrow keys and `w`, `a`, `s`, `d` as controls.

## Training
There are four training scripts for

* manually training a PPO agent on custom rewards (`ppo_train.py`),
* training AIRL on a single expert dataset (`airl_train.py`),
* active MORL with custom/automatic preferences (`moral_train.py`) and
* training DRLHP with custom/automatic preferences (`drlhp_train.py`).

When using automatic preferences, a desired ratio can be passed as an argument. For example, 

  ``python moral_train.py --ratio a b c``

will run MORAL using a (real-valued) ratio of `a:b:c` among the three explicit objectives in Delivery.


## Hyperparameters
Hyperparameters are passed as arguments to `wandb.init()` and can be changed by modifying the respective training files.
