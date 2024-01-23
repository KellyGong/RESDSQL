import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

warnings.filterwarnings("ignore")

model_name = "/data/gongzheng/llm/deepseek-moe-16b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda:1")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

# messages = [
#     {"role": "user", "content": "There are a database has several tables: head : head.age , head.born_state , head.name , head.head_id | department : department.name , department.department_id , department.budget_in_billions , department.creation , department.num_employees | management : management.head_id , management.temporary_acting , management.department_id | management.head_id = head.head_id | management.department_id = department.department_id, can you give me the SQL about How many heads of the departments are older than 56 ?"}
# ]
# input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
# outputs = model.generate(input_tensor.to(model.device), max_new_tokens=200)

# result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
# print(result)

import json

with open("data/preprocessed_data/resdsql_dev.json") as f:
    eval_data = json.load(f)

# with open("data/output/deepseek-moe-16b-chat-fail.json", 'r') as f:
#     eval_data = f.readlines()

# input_sequences = [eval_data[i] for i in range(len(eval_data)) if i % 2 == 0]
# output_sqls = [eval_data[i] for i in range(len(eval_data)) if i % 2 == 1]

input_sequences = [x['input_sequence'] for x in eval_data]
output_sqls = [x['output_sequence'].split('|')[1] for x in eval_data]


from tqdm import tqdm
import re

fail_question = []
fail_sql = []

with open("data/output/deepseek-moe-16b-chat.json", "a+") as f:
    for input_seq, output_sql in tqdm(zip(input_sequences, output_sqls), total=len(input_sequences)):
        try:
            question, schema = input_seq.split('|', 1)
            messages = [{"role": "user", "content": f"There are a database has several tables: {schema}, can you give me the SQL about {question}"}]
            input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            outputs = model.generate(input_tensor.to(model.device), max_new_tokens=250)
            result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
            result_sql = re.findall(r"```sql(.*)```", result, re.DOTALL)
            if len(result_sql) != 0:
                result_sql = result_sql[0].replace('\n', ' ').replace(';', ' ')
            else:
                result_sql = re.findall(r"```(.*)```", result, re.DOTALL)[0].replace('\n', ' ').replace(';', ' ')
            if "```" in result_sql:
                result_sql = result_sql.split("```")[0]

            f.write(result_sql + '\n')
            f.write(output_sql + '\n')
        except Exception as e:
            f.write('fail' + '\n')
            f.write(output_sql + '\n')
            fail_question.append(input_seq)
            fail_sql.append(output_sql)
            
with open("data/output/deepseek-moe-16b-chat-fail.json", "a+") as f:
    for input_seq, output_sql in tqdm(zip(fail_question, fail_sql)):
        f.write(input_seq + '\n')
        f.write(output_sql + '\n')

