import sys
from pathlib import Path
from transformers import pipeline, GenerationConfig, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from conversation import ConversationContext
from mps_index import MPSVectorIndex
import textwrap

# Use MPS for MacBook GPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load models
print("Loading models...")
llm = pipeline('text-generation', model='Qwen/Qwen2.5-3B-Instruct', device=device)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')

MAX_CONTEXT_TOKENS = 32000
MAX_GENERATION_TOKENS = 1024

def load_skill(skill_path):
    """Load skill from SKILL.md file"""
    try:
        with open(skill_path, 'r') as f:
            return f.read()
    except:
        return ""

# Load Python expert skill
SKILL = load_skill('skills/python-expert/SKILL.md')

def chunk_text(text, max_tokens=500, overlap_tokens=50):
    """Split text into token-aware chunks with overlap"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        
        if end >= len(tokens):
            break
        start = end - overlap_tokens
    
    return chunks

def read_file(file_path):
    """Read any text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return None

def create_index(text):
    """Create MPS vector index from text chunks"""
    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks)
    
    index = MPSVectorIndex(device=device)
    index.add(embeddings)
    
    return index, chunks

def count_tokens(text):
    """Count tokens in text"""
    return len(tokenizer.encode(text, add_special_tokens=False))

def search_chunks(index, chunks, question, max_chunks=10):
    """Find most relevant chunks within token budget"""
    question_embedding = embedder.encode([question])
    distances, indices = index.search(question_embedding, k=max_chunks)
    
    # Select chunks that fit within token budget
    selected_chunks = []
    total_tokens = count_tokens(question) + 200  # Question + prompt overhead
    
    for i in indices[0]:
        chunk = chunks[i]
        chunk_tokens = count_tokens(chunk)
        
        if total_tokens + chunk_tokens < (MAX_CONTEXT_TOKENS - MAX_GENERATION_TOKENS):
            selected_chunks.append(chunk)
            total_tokens += chunk_tokens
        else:
            break
    
    return selected_chunks if selected_chunks else [chunks[indices[0][0]]]

def ask_question(relevant_chunks, question, conversation):
    """Ask question with relevant context"""
    context = "\n\n".join(relevant_chunks)
    
    skill_context = f"\n\n{SKILL}\n\n" if SKILL else ""
    
    prompt = f"""You are a helpful assistant. Answer questions about the provided content concisely and accurately.
{skill_context}
Content:
{context}
{conversation.get_context()}

Question: {question}

Answer:"""
    
    gen_config = GenerationConfig(
        max_new_tokens=MAX_GENERATION_TOKENS,
        do_sample=False,
        pad_token_id=llm.tokenizer.eos_token_id,
        eos_token_id=llm.tokenizer.eos_token_id
    )
    
    result = llm(prompt, generation_config=gen_config, return_full_text=False)
    return result[0]['generated_text'].strip()

def main():
    if len(sys.argv) < 2:
        print("Usage: python file_qa.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    conversation = ConversationContext(max_history=5)
    
    print(f"\nReading file: {file_path}")
    content = read_file(file_path)
    
    if not content:
        print("Could not read file")
        sys.exit(1)
    
    print(f"Creating index from {len(content)} characters...")
    index, chunks = create_index(content)
    print(f"Created {len(chunks)} chunks")
    
    print("\nReady! Ask questions about the file")
    print("Commands: 'exit' to quit, 'load <path>' to change file, 'clear' to reset conversation\n")
    
    while True:
        question = input("Q: ").strip()
        
        if question.lower() in ['exit', 'quit', 'q']:
            break
        
        if question.lower() == 'clear':
            conversation.clear()
            print("Conversation history cleared\n")
            continue
        
        if question.lower().startswith('load '):
            new_file = question[5:].strip()
            print(f"\nReading file: {new_file}")
            new_content = read_file(new_file)
            if new_content:
                content = new_content
                file_path = new_file
                conversation.clear()
                print(f"Creating index from {len(content)} characters...")
                index, chunks = create_index(content)
                print(f"Created {len(chunks)} chunks\n")
            else:
                print("Could not read file\n")
            continue
        
        if not question:
            continue
        
        relevant_chunks = search_chunks(index, chunks, question)
        answer = ask_question(relevant_chunks, question, conversation)
        conversation.add(question, answer)
        
        print("\nA:", answer, "\n")

if __name__ == "__main__":
    main()
