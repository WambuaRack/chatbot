import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize the model and tokenizer
model_name = 'gpt2'  # You can change this to 'gpt2-medium', 'gpt2-large', etc., if needed
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Define the input context
input_context = (
    "The U.S. Department of Education has released a report on the federal government's efforts "
    "to improve education for children. The report, titled \"The U.S. Education System's Efforts "
    "to Improve Education for Children,\" is the first in a series of reports on the federal government's "
    "efforts to improve education for children."
)

# Encode the input context
input_ids = tokenizer.encode(input_context, return_tensors='pt')

# Generate output with adjusted parameters
output = model.generate(
    input_ids, 
    max_length=150, 
    num_return_sequences=1, 
    no_repeat_ngram_size=2, 
    temperature=0.7, 
    top_p=0.9, 
    top_k=50
)

# Decode the generated tokens
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Response:", output_text)
