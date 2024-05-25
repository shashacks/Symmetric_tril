
algo='ppo' # we add the reverse RL loss on PPO
robust_alpha=0.5 # when robust_alpha=1.0 and robust_beta=0.5, it's normal ppo
robust_beta=0.2

experiment_base_name="tril_experiment/tldr_output/${algo}"

for seed in {42..46}; do
    accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --num_processes 1\
        main.py task=tldr alg=ppo alg.tldr.args.seed=$seed alg.tldr.args.batch_size=16 alg.tldr.args.trajectories_per_update=64 sampling.batch_size_per_process=16 alg.tldr.args.save_every=50\
        alg.tldr.args.robust_beta=$robust_beta alg.tldr.args.robust_alpha=$robust_alpha \
        experiment_name="${experiment_base_name}_a${robust_alpha}_b${robust_beta}/${seed}"
end