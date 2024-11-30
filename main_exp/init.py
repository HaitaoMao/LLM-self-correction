from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer
import torch
# Initialize model and tokenizer
LLAMA_7B_DIR = "/scratch0/liuguan5/llama/llama-2-7b-chat-hf/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def init_model(args):
    """
    Initializes the model and tokenizer based on the provided arguments.
    """
    if args.llm == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(LLAMA_7B_DIR)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = LlamaForCausalLM.from_pretrained(LLAMA_7B_DIR, device_map='auto', output_attentions=True, return_dict=True)
        return tokenizer, model

    elif args.llm == "llama3":
        tokenizer = AutoTokenizer.from_pretrained("mlabonne/OrpoLlama-3-8B", cache_dir="/data/bochuan/DPO/cache")
        model = AutoModelForCausalLM.from_pretrained("mlabonne/OrpoLlama-3-8B", cache_dir="/data/bochuan/DPO/cache")
        return tokenizer, model.to(DEVICE)

    elif args.llm == "mistral":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", cache_dir="/data/bochuan/DPO/cache")
        model = AutoModelForCausalLM.from_pretrained(args.name_or_path, local_files_only=True, cache_dir="/data/bochuan/DPO/cache")
        return tokenizer, model.to(DEVICE)
