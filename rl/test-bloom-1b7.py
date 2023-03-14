from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b7")

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b7")
