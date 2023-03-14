# https://towardsdatascience.com/getting-started-with-bloom-9e3295459b65
import torch
import transformers
from transformers import BloomForCausalLM, BloomTokenizerFast

model = BloomForCausalLM.from_pretrained("bigscience/bloom-1b7")
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-1b7")

prompt = "It was a dark and stormy night"
result_length = 50
inputs = tokenizer(prompt, return_tensors="pt")

# Greedy Search
# 2 minutes, 22 seconds
print(
    tokenizer.decode(model.generate(inputs["input_ids"], max_length=result_length)[0])
)

# Beam Search
print(
    tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            max_length=result_length,
            num_beams=2,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )[0]
    )
)

# Sampling Top-k + Top-p
print(
    tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            max_length=result_length,
            do_sample=True,
            top_k=50,
            top_p=0.9,
        )[0]
    )
)
