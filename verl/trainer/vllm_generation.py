from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn import functional as F
import argparse
import json
import sys
import os
from vllm import LLM
# Reduce VRAM usage by reducing fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


import os
def ensure_dir_for_file(file_path):
    """
    检查给定的文件路径中的目录是否存在，不存在则创建目录。
    
    参数:
        file_path (str): 要保存的文件的完整路径。
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"目录已创建：{directory}")
    else:
        print(f"目录已存在：{directory}")

class VLLMGeneration:
    def __init__(self, 
        config: dict,
        model_path: str
    ):
        self.config = config
        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)
        max_model_len = int(self.config.max_model_len or self.config.prompt_length + self.config.response_length)
        load_format = "dummy" if self.config.load_format.startswith("dummy") else self.config.load_format
        engine_kwargs = self.config.get("engine_kwargs", {}).get("vllm", {}) or {}
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            max_num_seqs=config.max_num_seqs,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=config.enable_prefix_caching,
            seed=config.get("seed", 0),
            **engine_kwargs,
        )
         

    
    def format_input(self, inputs):
        # print("inputs", inputs)
        if "messages" in inputs:
            messages = inputs['messages']
        else:
            
            instruction, input, output = inputs['instruction'], inputs['input'], inputs['output']
            messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input},
            {"role": "assistant", "content": output},
            ]
        tok_only_query = self.tok.apply_chat_template(messages, tokenize=True, return_tensors="pt", max_length=self.max_length, truncation=True)
       
     

        return tok_only_query


    def tokenize(self, input):
        # print("input", input)
        tok_only_query, input_mask, output_mask, labels = self.format_input(input)
        
        input_len = torch.sum(input_mask[0]==1)
        output_len = torch.sum(input_mask[0]!=1)
        e = {
            "input_ids": tok_only_query,
            "input_mask": input_mask,
            "output_mask": output_mask,
            "label": labels
        }
        return e



    def get_score(self, input):
        input_mask = input['input_mask']
        output_mask = input['output_mask']
        # activation_source = activation_source[:, choice_start-1:choice_end-1]
        # activation_target = activation_target[:, choice_start-1:choice_end-1]
        neuron_total = torch.cat(self.activations, dim=0)
        print("neuron_total", neuron_total.shape)
        # print("neuron_total", neuron_total.shape)
        # neuron_input = extract_by_mask(neuron_total, input_mask)
        # # print("neuron_input", neuron_input.shape)
        # neuron_input = torch.mean(neuron_total, dim=1)
        # # print("neuron_input", neuron_input.shape)
        # neuron_input = neuron_input.view(1, -1).squeeze(0)
        # target_ac = neuron_input[self.target_neurons]
        # neuron_total =torch.mean(neuron_total, dim=1)
        # neuron_total = neuron_total.view(1, -1).squeeze(0)
        del self.activations
        self.activations = []
        return neuron_total.cpu()
            



    def get_rpe(self, inputs, i):
        # Tokenize the text

        inputs_ori = self.tokenize(inputs)


        # sys.exit(0)
        # Get the logits
        with torch.no_grad():
            input_ids=inputs_ori['input_ids'].to(device)
            outputs_ori = self.source_model(
                input_ids=input_ids,
                labels=inputs_ori['label'].to(device),
                return_dict=True  
                )
        loss = outputs_ori.loss.item()
        hidden_states = outputs_ori.hidden_states
        target_ac = self.get_score(inputs_ori)
        self.target_neuron_activation[self.start_idx + i] = target_ac
        self.loss[self.start_idx + i] = loss
        torch.cuda.empty_cache()
        return target_ac
     
def load_data(data_path):
    if data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            data = json.load(f)
    elif data_path.endswith('.jsonl'):
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    return data
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--model_name_or_path_source", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=0)
    parser.add_argument("--neuron_file", type=str, required=True)
    parser.add_argument("--prompt", type=str, default='wiz', help='wiz, alpaca')

 
    args = parser.parse_args()
    return args


from tqdm import tqdm
def main():

    args = parse_args()
    tdns = TrainDataNeuronSelection(args)

    data = load_data(args.data_path)
    if args.end_idx == 0:
        args.end_idx = len(data)
    # data = data[:33838]
    data_sample = data[args.start_idx:args.end_idx]
    
    for i in tqdm(range(len(data_sample)), desc='Calculating activations'):
        input = data_sample[i]
        # if input.get('examples') is not None:
            # if args.icl == False:
            #     input.pop('examples')
        target_ac = tdns.get_rpe(input, i)

    activations = tdns.target_neuron_activation
    ensure_dir_for_file(args.save_path)
    torch.save(activations, args.save_path)
    
    

    

if __name__ == '__main__':
    main()
    