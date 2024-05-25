#!/bin/bash
accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --num_processes 1 eval.py\
 task=tldr alg=ppo alg.tldr.args.batch_size=2 alg.tldr.args.trajectories_per_update=8\
 sampling.batch_size_per_process=2 alg.tldr.args.n_iters=100