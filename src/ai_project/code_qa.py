import os
import sys
from pathlib import Path
from transformers import pipeline, GenerationConfig
import torch
from conversation import ConversationContext

# Use MPS for MacBook GPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
print("Loading model...")
generator = pipeline('text-generation', model='Qwen/Qwen2.5-3B-Instruct', device=device)


def read_code_files(directory, extensions=None):
    """Read all code files from directory"""
    if extensions is None:
        extensions = ['.py', '.js', '.java', '.scala', 'sql']
    code_content = []
    dir_path = Path(directory)
    
    if not dir_path.exists():
        return None
    
    for ext in extensions:
        for file_path in dir_path.rglob(f'*{ext}'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    code_content.append(f"File: {file_path.relative_to(dir_path)}\n{content}\n")
            except:
                continue
    
    return "\n".join(code_content)

def ask_question(context, question, conversation):
    """Ask question about code with conversation history"""
    prompt = f"""
You are a code analysis assistant. Answer questions about the provided code.

Code:
{context[:3000]}
{conversation.get_context()}

Question: {question}

Answer:
"""
    
    gen_config = GenerationConfig(
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    
    result = generator(prompt, generation_config=gen_config, return_full_text=False)
    
    return result[0]['generated_text'].strip()

def main():
    if len(sys.argv) < 2:
        print("Usage: python code_qa.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    conversation = ConversationContext(max_history=5)
    
    print(f"\nReading code from: {directory}")
    context = read_code_files(directory)
    
    if not context:
        print("No code files found or directory doesn't exist")
        sys.exit(1)
    
    print(f"Loaded {len(context)} characters of code")
    print("\nReady! Ask questions about the code")
    print("Commands: 'exit' to quit, 'load <path>' to change directory, 'clear' to reset conversation\n")
    
    while True:
        question = input("Q: ").strip()
        
        if question.lower() in ['exit', 'quit', 'q']:
            break
        
        if question.lower() == 'clear':
            conversation.clear()
            print("Conversation history cleared\n")
            continue
        
        if question.lower().startswith('load '):
            new_dir = question[5:].strip()
            print(f"\nReading code from: {new_dir}")
            new_context = read_code_files(new_dir)
            if new_context:
                context = new_context
                directory = new_dir
                conversation.clear()
                print(f"Loaded {len(context)} characters of code\n")
            else:
                print("No code files found or directory doesn't exist\n")
            continue
        
        if not question:
            continue
        
        answer = ask_question(context, question, conversation)
        conversation.add(question, answer)
        print("\nA:", answer, "\n")

if __name__ == "__main__":
    main()
