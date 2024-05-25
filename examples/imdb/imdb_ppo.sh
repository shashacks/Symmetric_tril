#!/bin/bash

algo='ppo' # we add the reverse RL loss on PPO
robust_alpha=1.0 
robust_beta=0.0

experiment_base_name="tril_experiment/imdb_output/${algo}"
for seed in {42..46}; do
    accelerate launch --config_file accelerate_cfgs/fsdp_config.yaml --num_processes 1 main.py task=imdb alg=ppo\
        alg.imdb.args.seed=$seed task.args.seed=$seed\
        alg.imdb.args.robust_beta=$robust_beta alg.imdb.args.robust_alpha=$robust_alpha \
        experiment_name="${experiment_base_name}_a${robust_alpha}_b${robust_beta}/${seed}"
end