import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from calflops import calculate_flops

from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


warnings.filterwarnings("ignore")

model_name = "/data/gongzheng/llm/deepseek-moe-16b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda:4")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

# messages = [
#     {"role": "user", "content": "There are a database has several tables: head : head.age , head.born_state , head.name , head.head_id | department : department.name , department.department_id , department.budget_in_billions , department.creation , department.num_employees | management : management.head_id , management.temporary_acting , management.department_id | management.head_id = head.head_id | management.department_id = department.department_id, can you give me the SQL about How many heads of the departments are older than 56 ?"}
# ]
# input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
# outputs = model.generate(input_tensor.to(model.device), max_new_tokens=200)

# result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
# print(result)

batch_size = 1
max_seq_length = 1024

flops, macs, params = calculate_flops(model, input_shape=(batch_size, max_seq_length), transformer_tokenizer=tokenizer)

print("deepseek-moe-16b-chat FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

input_seq = 'How many heads of the departments are older than 56 ? | head : head.age , head.born_state , How many heads of the departments are older than 56 ? | head : head.age , head.born_state ,How many heads of the departments are older than 56 ? | head : head.age , head.born_state ,How many heads of the departments are older than 56 ? | head : head.age , head.born_state ,'

question, schema = input_seq.split('|', 1)
messages = [{"role": "user", "content": f"There are a database has several tables: {schema}, can you give me the SQL about {question}"}]
input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", padding=True, max_length=max_seq_length)
outputs = model.generate(input_tensor.to(model.device), max_new_tokens=200)
result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)

# 检查模型显存使用情况
memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 3)  # 显存占用大小（以GB为单位）
print(f"模型显存使用情况：{memory_usage:.2f} GB")

print_gpu_utilization()
