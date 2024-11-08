import os
from transformers import AutoModelForCausalLM, AutoModel

def download_model(model_name, save_path):
    os.makedirs(save_path, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=save_path)
    print(f"Model {model_name} downloaded to {save_path}")


save_path = "checkpoints/"
model_name = "meta-llama/Llama-2-7b-chat-hf"
download_model(model_name, save_path)