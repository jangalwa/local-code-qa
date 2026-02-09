from transformers import pipeline, GenerationConfig
import torch

# Use MPS (Metal Performance Shaders) for MacBook GPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load a small model that runs locally on Mac GPU
generator = pipeline('text-generation', model='Qwen/Qwen2.5-3B-Instruct', device=device)

# Define your skill/prompt
skill = """You are a git commit message expert. Write clear, conventional commit messages.

Format: <type>: <description>
Types: feat, fix, docs, style, refactor, test, chore

Example: feat: add user authentication"""

# Use it
prompt = f"{skill}\n\nTask: Write a commit message for adding a login feature\n\nCommit:"

gen_config = GenerationConfig(
    max_new_tokens=50,
    num_return_sequences=1,
    do_sample=False,
    pad_token_id=generator.tokenizer.eos_token_id
)

result = generator(prompt, generation_config=gen_config, return_full_text=False)
print("\n\n=======\n\nGenerated commit message: ")
print(result[0]['generated_text'])
