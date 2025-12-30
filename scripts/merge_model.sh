python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir /local_data/shares/yhe/checkpoints/replay-grpo-epoch4/Qwen3-8B-deepscaler-NODE1-2/global_step_100/actor \
    --target_dir /local_data/shares/yhe/checkpoints/replay-grpo-epoch4/Qwen3-8B-deepscaler-NODE1-2/100_model \
    --hf_config_path /local_data/shares/yhe/checkpoints/replay-grpo-epoch4/Qwen3-8B-deepscaler-NODE1-2/global_step_100/actor/huggingface/