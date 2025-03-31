from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "deepseek-ai/deepseek-llm-7b-chat"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained("./deepseek-llm-7b-chat")
tokenizer.save_pretrained("./deepseek-llm-7b-chat")
