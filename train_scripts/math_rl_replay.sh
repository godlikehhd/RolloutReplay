set -x
WORKING_DIR=verl
WORKING_ROOT_DIR=$(dirname $WORKING_DIR)
echo pwd: $(pwd)
unset VLLM_USE_MODELSCOPE LMDEPLOY_USE_MODELSCOPE

export WANDB_MODE=offline
export WANDB_DIR=/nfsdata/yhe/wandb_logs
export WANDB_API_KEY=16b5bad0d1c1d68b959a024003969f9620561990


GPU_NUMS=8
WORLD_SIZE=2

DATA_NAME=deepscaler
# TRAIN_DATA_PATH="[$(echo ${WORKING_ROOT_DIR}/data/qwen-math/math-level3to5/train.parquet)]"
TRAIN_DATA_PATH="[$(echo ${WORKING_ROOT_DIR}/data/qwen-math/deepscaler/train.parquet)]"
TEST_DATA_PATH="[$(echo ${WORKING_ROOT_DIR}/data/eval-data-no-system-prompt/aime24x10.parquet),$(echo ${WORKING_ROOT_DIR}/data/eval-data-no-system-prompt/aime25x10.parquet),$(echo ${WORKING_ROOT_DIR}/data/eval-data-no-system-prompt/amc23.parquet),$(echo ${WORKING_ROOT_DIR}/data/eval-data-no-system-prompt/math500.parquet),$(echo ${WORKING_ROOT_DIR}/data/eval-data-no-system-prompt/olympiadbench.parquet)]"

MODEL_NAME=DeepSeek-R1-Distill-Qwen-1.5B
MODEL_PATH=/nfsdata/yhe/models/DeepSeek-R1-Distill-Qwen-1.5B

REWARD_FN_PATH=${WORKING_ROOT_DIR}/verl/utils/reward_score/math_verification.py

EXP_NOTE=test-2k-no-entctl-${VERIFICATION_REWARD_TYPE}

PROJECT_NAME=replay-verl-grpo2
EXPERIMENT_NAME=${EXP_NOTE}-${MODEL_NAME}-${DATA_NAME}-NODE${WORLD_SIZE}
OUTPUT_DIR=checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
mkdir -p ${OUTPUT_DIR}

PROMPT_LENGTH=2048
RESPONSE_LENGTH=6144
NUM_BATCHED_TOKENS=$((${PROMPT_LENGTH} + ${RESPONSE_LENGTH}))
MAX_TOKEN_PER_GPU=$(((${PROMPT_LENGTH} + ${RESPONSE_LENGTH}) * 2))

VERIFICATION_REWARD_TYPE=fix_imbalance
AUXILIARY_REWARDS="'non_short_response,no_code'"


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_DATA_PATH} \
    data.val_files=${TEST_DATA_PATH} \
    data.train_batch_size=64 \
    data.max_prompt_length=${PROMPT_LENGTH} \
    data.max_response_length=${RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    +actor_rollout_ref.rollout.logprobs=10 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=${NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward_model.reward_manager=naive \
    custom_reward_function.path=${REWARD_FN_PATH} \
    reward_config.verification_reward_type=${VERIFICATION_REWARD_TYPE} \
    reward_config.auxiliary_rewards=${AUXILIARY_REWARDS} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.log_val_generations=10 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=${GPU_NUMS} \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.total_epochs=30 \
    trainer.total_training_steps=500 \
    trainer.save_online_data.enabled=True \
    trainer.save_online_data.train_data_size=40000 \
    trainer.save_online_data.val_data_size=200 \
    trainer.save_online_data.online_save_freq=50 \
    trainer.use_reward_shaping=True \
    trainer.do_replay=True \
    2>&1 | tee ${OUTPUT_DIR}/train.log

# cd ${WORKING_ROOT_DIR}
# python scripts/model_utils/convert_final_ckpt.py ${OUTPUT_DIR}
