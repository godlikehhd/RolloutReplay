set -x



# export VERL_LOGGING_LEVEL=DEBUG
# export CUDA_VISIBLE_DEVICES=0


data_path=/home/heye/Efficient-RL/data/qwen-math/deepscaler/train_50.parquet
save_path=/home/heye/Efficient-RL/analysis_data/train_50_generation_math_instruct_test.parquet
model_path=/data/groups/QY_LLM_Other/sq_models/Qwen/Qwen2.5-0.5B-Instruct


python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=8 \
    data.output_path=$save_path \
    model.path=$model_path \
    +model.trust_remote_code=True \
    rollout.name=vllm \
    rollout.temperature=1.0 \
    rollout.calculate_log_probs=False \
    rollout.prompt_length=1024 \
    rollout.response_length=4096 \
    rollout.top_k=-1 \
    rollout.top_p=1.0 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8