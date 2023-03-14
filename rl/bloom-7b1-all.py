# https://towardsdatascience.com/getting-started-with-bloom-9e3295459b65
import sys

import torch
import transformers
from transformers import BloomForCausalLM, BloomTokenizerFast

model = BloomForCausalLM.from_pretrained("bigscience/bloom-7b1")
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-7b1")

prompt = sys.argv[1]
result_length = 50
inputs = tokenizer(prompt, return_tensors="pt")

# Greedy Search
print(
    tokenizer.decode(model.generate(inputs["input_ids"], max_length=result_length)[0])
)
print("----------------------------------------")

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
print("----------------------------------------")

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
