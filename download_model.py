import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "deepseek-ai/deepseek-llm-7b-chat"
model_path = "./deepseek-chat"

# 下载并保存模型
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

print("模型下载完成！")
