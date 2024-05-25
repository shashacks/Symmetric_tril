# Symmetric Reinforcement Learning Loss for Robust Learning on Diverse Tasks and Model Scales

Reinforcement learning (RL) training is inherently unstable due to factors such as moving targets and high gradient variance. Reinforcement Learning from Human Feedback (RLHF) and Reinforcement Learning from AI Feedback (RLAIF) can introduce additional difficulty. Differing preferences can complicate the alignment process, and prediction errors in a trained reward model can become more severe as the LLM generates unseen outputs. To enhance training robustness, RL has adopted techniques from supervised learning, such as ensembles and layer normalization. In this work, we improve the stability of RL training by adapting the reverse cross entropy (RCE) from supervised learning for noisy data to define a symmetric RL loss. We demonstrate performance improvements across various tasks and scales. We conduct experiments in discrete action tasks (Atari games) and continuous action space tasks (MuJoCo benchmark and Box2D) using Symmetric A2C (SA2C) and Symmetric PPO (SPPO), with and without added noise with especially notable performance in SPPO across different hyperparameters. Furthermore, we validate the benefits of the symmetric RL loss when using SPPO for large language models through improved performance in RLHF tasks, such as IMDB positive sentiment sentiment and TL;DR summarization tasks. 

We implement our method based on [TRIL](https://github.com/Cornell-RL/tril/tree/main). This repository is for IMDB positive sentiment analysis and TL;DR summarization. We add the advantage normalization and the symmetric RL part to it. For Atari games, MuJoCo benchmark, and Box2D. Please refer to LLM tasks [here](https://github.com/shashacks/Symmetric_RL). 



## Installation
Note that we use accelerate=0.27.2 which is different from the original code to solve an error.
```
conda create -n tril python=3.10
conda activate tril
pip install -e .
```

## Example Scripts
To run SPPO for IMDB positive sentiment
```
./examples/imdb/imdb_sppo.sh
```

To run PPO for IMDB positive sentiment
```
./examples/imdb/imdb_ppo.sh
```

To run SPPO for TL;DR summarization
```
./examples/tldr/tldr_sppo.sh
```

To run PPO for TL;DR summarization
```
./examples/tldr/tldr_ppo.sh
```

## Evaluation for TL;DR
We follow [TRIL](https://github.com/Cornell-RL/tril/tree/main), where they evaluate their models' perplexity after training. Here, we provide the script for the evaluation. For the perplexity metric, you need to comment in and out of the script cfgs/task/tldr.yaml (please see the script).
```
./examples/tldr/tldr_eval.sh
```




<!-- ## Usage Example
Here is a minimal example of running PPO with TRIL:
```python
import hydra
from accelerate import Accelerator
from tril import tril_run
from tril.logging import Tracker
from tril.algorithms import PPO

@hydra.main(version_base=None, config_path="cfgs", config_name="config") # Hydra Decorator for Config
@tril_run # TRIL decorator for hydra config processing
def run_ppo(cfg):
    # Initialize accelerator for distributed computing
    accelerator = Accelerator()

    # Grab experiment save directory from Hydra
    save_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Instantiate TRIL logger for WandB and CLI logging/saving
    tracker = Tracker(
        save_path,
        OmegaConf.to_container(cfg, resolve=True),
        cfg.project_name,
        cfg.experiment_name,
        cfg.entity_name,
        cfg.log_to_wandb,
        log_level=logging.INFO,
        is_main_process=accelerator.is_main_process,
    )

    # Instantiate Algorithm
    ppo = PPO(cfg, accelerator, tracker)

    # Start learn to train LLM
    ppo.learn()

if __name__ == '__main__':
    run_ppo()
```

`TRIL` also provides an [`AlgorithmRegistry`](https://github.com/Cornell-RL/tril/blob/main/src/tril/algorithms/__init__.py) to instantiate algorithms. Please see our `main.py` to see how our scripts instantiate the algorithms. The list of available algorithms can be seen by the configs in `cfgs/task`.

## Current Task/Algorithm Support Matrix

| Algorithm  | IMDB | CommonGen | TL;DR |
|------------| ---- | ---- | ---- |
| PPO        | ✅ | ✅ | ✅ |
| PPO++      | ✅ | ✅ | ✅ |
| AggreVaTeD | ✅ | ✅ | ✅ |
| LOLS       | ✅ | ✅ | ✅ |
| D2LOLS     | ✅ | ✅ | ✅ |
| BC         | ✅ | ✅ | ✅ |
| GAIL       | ✅ |  |  |

## Code Structure
The directory structure of the configs, run script, and TRIL components looks like this.

```
├── cfgs                    <- Hydra configs
│   ├── alg                 <- Algorithm configs (e.g. PPO)
│   ├── task                <- Task configs (e.g. TL;DR summarization)
│   ├── logging             <- Logging configs (e.g. WandB)
│   │
│   └── config.yaml         <- Main config for training
│
├── accelerate_cfgs         <- Accelerate configs
│
├── main.py                 <- TRIL main function
│
├── tril                    <- TRIL src
│   ├── algorithms          <- Algorithm implementations
│   ├── buffers             <- Data Buffer (e.g. OnlineBuffer, PromptBuffer)
│   ├── metrics             <- Evaluation Metrics
│   ├── policies            <- Language Model Policies (e.g. Actor, ActorCritic)
│   ├── rewards             <- Reward Functions
│   ├── tasks               <- Supported Tasks
│   ├── utils               <- Helper functions for TRIL
│   │
│   ├── agent.py            <- Agent contains all torch.nn Modules (i.e. Policy and Reward)
│   ├── base_algorithm.py   <- Algorithm abstract class
│   ├── base_metric.py      <- Metric abstract class
│   ├── base_reward.py      <- Reward abstract class
│   ├── base_task.py        <- Task abstract class
│   └── logging.py          <- TRIL Logger
```

In each directory's `__init__.py`, there is a registry to register all supported `algorithms`, `metrics`, `rewards`, and `tasks`. When extending `TRIL`, please add the respective addition to one of these registries.

## Logging
TRIL support Weights and Biases logging. Please enter your `wandb` details such as `entity_name` and `project_name` into `cfgs/logging/wandb.yaml`. If you would not like to log to `wandb`, please set `log_to_wandb=False`.

By default, we save training and evaluation information in `outputs/<experiment_name>/<datetime>` You can define `experiment_name` in `cfgs/config.yaml` or through Hydra CLI, `main.py experiment_name=<name>`.


## Example WandB Reports
Here is an example WandB Report of how the logging would look like when running multiple different algorithms

* [CommonGen Report](https://api.wandb.ai/links/coactivelearning/hfocjp17).
* [TL;DR PPO Report](https://api.wandb.ai/links/coactivelearning/ga4r1uqd).

## Citing TRIL
If you use TRIL in your publication, please cite it by using the following BibTeX entry.
```bibtex
@misc{TRIL,
      title={TRIL: Transformers Reinforcement and Imitation Learning Library},
      author={Jonathan D Chang and Kiante Brantley and Rajkumar Ramamurthy and Dipendra Misra and Wen Sun},
      howpublished={\url{https://github.com/Cornell-RL/tril}},
      year={2023}
}
```

Here is the citation of the accompanying paper for many of the supported algorithms.
```bibtex
@misc{chang2023learning,
      title={Learning to Generate Better Than Your LLM}, 
      author={Jonathan D. Chang and Kiante Brantley and Rajkumar Ramamurthy and Dipendra Misra and Wen Sun},
      year={2023},
      eprint={2306.11816},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgements
We would like to acknowledge [RL4LMs](https://github.com/allenai/RL4LMs), [TRL](https://github.com/huggingface/trl), and [TRLx](https://github.com/CarperAI/trlx) for being inspirations for this library. -->
