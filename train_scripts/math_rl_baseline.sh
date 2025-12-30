set -x

cd /home/yhe/verl   # 确保进入项目根目录
WORKING_ROOT_DIR=$(pwd)
echo pwd: $(pwd)
unset VLLM_USE_MODELSCOPE LMDEPLOY_USE_MODELSCOPE

export WANDB_MODE=offline
export WANDB_DIR=/home/yhe/wandb_logs
export WANDB_API_KEY=16b5bad0d1c1d68b959a024003969f9620561990


GPU_NUMS=8
WORLD_SIZE=1

DATA_NAME=deepscaler


math500_test_path=${WORKING_ROOT_DIR}/data/math500/test.parquet
amc23_test_path=${WORKING_ROOT_DIR}/data/amc23/test.parquet
amc24_test_path=${WORKING_ROOT_DIR}/data/amc24/test.parquet
aime2024_test_path=${WORKING_ROOT_DIR}/data/aime2024x10/test.parquet
aime2025_test_path=${WORKING_ROOT_DIR}/data/aime2025x10/test.parquet
gaokao_test_path=${WORKING_ROOT_DIR}/data/gaokao/test.parquet
minerva_test_path=${WORKING_ROOT_DIR}/data/minervamath/test.parquet
olympiad_test_path=${WORKING_ROOT_DIR}/data/olympiadbench/test.parquet

test_files="['$aime2024_test_path', '$olympiad_test_path', '$aime2025_test_path']"


# TRAIN_DATA_PATH="[$(echo ${WORKING_ROOT_DIR}/data/qwen-math/math-level3to5/train.parquet)]"
TRAIN_DATA_PATH="[$(echo ${WORKING_ROOT_DIR}/data/qwen-math/deepscaler/train_800_dapo.parquet)]"
# TEST_DATA_PATH="[$(echo ${WORKING_ROOT_DIR}/data/eval-data-no-system-prompt/aime24x10.parquet),$(echo ${WORKING_ROOT_DIR}/data/eval-data-no-system-prompt/aime25x10.parquet),$(echo ${WORKING_ROOT_DIR}/data/eval-data-no-system-prompt/amc23.parquet),$(echo ${WORKING_ROOT_DIR}/data/eval-data-no-system-prompt/math500.parquet),$(echo ${WORKING_ROOT_DIR}/data/eval-data-no-system-prompt/olympiadbench.parquet)]"

MODEL_NAME=Qwen3-8B
MODEL_PATH=/local_data/shares/models/models/Qwen/Qwen3-8B




PROJECT_NAME=replay-grpo-epoch4
EXPERIMENT_NAME=${MODEL_NAME}-${DATA_NAME}-NODE${WORLD_SIZE}-baseline-online-8gpus-2
OUTPUT_DIR=/local_data/shares/yhe/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
save_batch_dir=${OUTPUT_DIR}/ppo_batchs
mkdir -p ${OUTPUT_DIR}
mkdir -p ${save_batch_dir}
PROMPT_LENGTH=1536 
RESPONSE_LENGTH=16384
NUM_BATCHED_TOKENS=$((${PROMPT_LENGTH} + ${RESPONSE_LENGTH}))
# export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA_PATH" \
    data.val_files="$test_files" \
    data.train_batch_size=8 \
    data.weighted_random_sampler=False \
    data.max_prompt_length=${PROMPT_LENGTH} \
    data.max_response_length=${RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.save_batch_dir=${save_batch_dir} \
    actor_rollout_ref.actor.compute_entropy=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.clip_ratio_low=0.6 \
    actor_rollout_ref.actor.clip_ratio_high=4.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.00 \
    actor_rollout_ref.actor.use_m2po_loss=False \
    actor_rollout_ref.actor.use_m2po_loss_symmetric=False \
    actor_rollout_ref.actor.M2_budget=0.04 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.max_num_batched_tokens=${NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_k=45 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.perturb_advantage=False \
    algorithm.normalize_batch_advantage=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.val_before_train=False \
    data.val_batch_size=512 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.do_replay=False \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.val_only=False \
    trainer.total_training_steps=401 \
    2>&1 | tee ${OUTPUT_DIR}/train.log

# cd ${WORKING_ROOT_DIR}
# python scripts/model_utils/convert_final_ckpt.py ${OUTPUT_DIR}
