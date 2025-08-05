import torch
import transformers, AutoTokenizer, AutoModelForCausalLM

# base Llama 3 8B Model
model_id = "meta-llama/Meta-Llama-3-8B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"            
)


