#!/bin/bash

algo='ppo' # we add the reverse RL loss on PPO
robust_alpha=0.5 # when robust_alpha=1.0 and robust_beta=0.5, it's normal ppo
robust_beta=0.4

experiment_base_name="tril_experiment/imdb_output/${algo}"
for seed in {42..46}; do
    accelerate launch --config_file accelerate_cfgs/fsdp_config.yaml --num_processes 1 main.py task=imdb alg=ppo\
        alg.imdb.args.seed=$seed task.args.seed=$seed\
        alg.imdb.args.robust_beta=$robust_beta alg.imdb.args.robust_alpha=$robust_alpha \
        experiment_name="${experiment_base_name}_a${robust_alpha}_b${robust_beta}/${seed}"
end