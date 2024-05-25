#!/bin/bash
accelerate launch --config_file accelerate_cfgs/fsdp_config.yaml --num_processes 1 main.py task=commongen alg=ppo alg.commongen.args.batch_size=4 sampling.batch_size_per_process=8
