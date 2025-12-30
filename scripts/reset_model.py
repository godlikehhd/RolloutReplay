from transformers import AutoModelForCausalLM
import gc

def merge_models(model_path, ref_model_path, torch_dtype, trust_remote_code, temp_path="outputs/temp_path", alpha=0.5, merged_name=None):# lm_head up_proj down_proj o_proj 
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)
    # print(ref_model)

    # 获取模型参数
    model_params = model.state_dict()
    ref_model_params = ref_model.state_dict()

    # 融合参数
    merged_params = {}
    for name in model_params.keys():
        if merged_name!=None and merged_name not in name:
            merged_params[name] = ref_model_params[name]
            continue
        else:
            theta_model = model_params[name]
            theta_ref_model = ref_model_params[name]
            merged_params[name] = alpha * theta_model + (1 - alpha) * theta_ref_model

    # 更新模型参数
    model.load_state_dict(merged_params)

    # 保存融合后的模型
    model.save_pretrained(temp_path)
    del model, ref_model, merged_params, model_params, ref_model_params
    gc.collect()
    print(f"The merged model has been saved to {temp_path}")

import torch
if __name__ == "__main__":
    model_path = "/local_data/shares/models/models/Qwen/Qwen3-8B"
    ref_model_path = "/local_data/shares/yhe/checkpoints/replay-grpo-epoch4/Qwen3-8B-deepscaler-NODE1-2/100_model"
    torch_dtype = torch.bfloat16
    trust_remote_code = True
    temp_path = "/local_data/shares/yhe/checkpoints/replay-grpo-epoch4/Qwen3-8B-deepscaler-NODE1-2/100_model_reset"
    alpha = 0.5
    merged_name = None
    merge_models(model_path, ref_model_path, torch_dtype, trust_remote_code, temp_path, alpha, merged_name)