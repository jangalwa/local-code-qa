import sys
import time
from pathlib import Path
from datetime import datetime
sys.path.insert(0, 'src')

from transformers import pipeline, GenerationConfig, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer
from ai_project.conversation import ConversationContext
from ai_project.mps_index import MPSVectorIndex

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load models
llm = pipeline('text-generation', model='Qwen/Qwen2.5-3B-Instruct', device=device)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')

def load_skill(skill_path):
    try:
        with open(skill_path, 'r') as f:
            return f.read()
    except:
        return ""

def ask_with_skill(context, question, skill=""):
    skill_context = f"\n\n{skill}\n\n" if skill else ""
    prompt = f"""You are a helpful assistant. Answer questions about the provided content concisely and accurately.
{skill_context}
Content:
{context}

Question: {question}

Answer:"""
    
    gen_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=True,
        temperature=0.1,
        pad_token_id=llm.tokenizer.eos_token_id,
        eos_token_id=llm.tokenizer.eos_token_id
    )
    
    start = time.time()
    result = llm(prompt, generation_config=gen_config, return_full_text=False)
    elapsed = time.time() - start
    
    answer = result[0]['generated_text'].strip()
    tokens = len(tokenizer.encode(answer))
    
    return answer, elapsed, tokens

def benchmark(code_file, questions):
    with open(code_file, 'r') as f:
        code = f.read()
    
    skill = load_skill('skills/python-expert/SKILL.md')
    
    print(f"\n{'='*70}")
    print(f"SKILL PERFORMANCE BENCHMARK")
    print(f"{'='*70}")
    print(f"File: {code_file}")
    print(f"Skill: {'Python Expert' if skill else 'None'}")
    print(f"{'='*70}\n")
    
    total_time_no = 0
    total_time_yes = 0
    total_tokens_no = 0
    total_tokens_yes = 0
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'─'*70}")
        print(f"QUESTION {i}: {question}")
        print(f"{'─'*70}\n")
        
        # Without skill
        answer_no, time_no, tokens_no = ask_with_skill(code, question)
        total_time_no += time_no
        total_tokens_no += tokens_no
        
        # With skill
        answer_yes, time_yes, tokens_yes = ask_with_skill(code, question, skill)
        total_time_yes += time_yes
        total_tokens_yes += tokens_yes
        
        # Calculate improvements
        time_diff = time_yes - time_no
        time_pct = (time_diff / time_no * 100) if time_no > 0 else 0
        token_diff = tokens_yes - tokens_no
        token_pct = (token_diff / tokens_no * 100) if tokens_no > 0 else 0
        
        print(f"📊 WITHOUT SKILL:")
        print(f"   Time: {time_no:.2f}s | Tokens: {tokens_no}")
        print(f"   {answer_no}\n")
        
        print(f"🎯 WITH SKILL:")
        print(f"   Time: {time_yes:.2f}s | Tokens: {tokens_yes}")
        print(f"   {answer_yes}\n")
        
        print(f"📈 IMPACT:")
        print(f"   Time:   {time_diff:+.2f}s ({time_pct:+.1f}%)")
        print(f"   Tokens: {token_diff:+d} ({token_pct:+.1f}%)")
        
        if time_pct > 10:
            print(f"   ⚠️  Significantly slower with skill")
        elif time_pct < -10:
            print(f"   ✅ Significantly faster with skill")
        else:
            print(f"   ➡️  Similar speed")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*70}")
    
    avg_time_diff = (total_time_yes - total_time_no) / len(questions)
    avg_time_pct = ((total_time_yes - total_time_no) / total_time_no * 100) if total_time_no > 0 else 0
    avg_token_diff = (total_tokens_yes - total_tokens_no) / len(questions)
    avg_token_pct = ((total_tokens_yes - total_tokens_no) / total_tokens_no * 100) if total_tokens_no > 0 else 0
    
    print(f"\nAverage per question:")
    print(f"  Time:   {avg_time_diff:+.2f}s ({avg_time_pct:+.1f}%)")
    print(f"  Tokens: {avg_token_diff:+.1f} ({avg_token_pct:+.1f}%)")
    
    print(f"\nTotal across {len(questions)} questions:")
    print(f"  Without skill: {total_time_no:.2f}s, {total_tokens_no} tokens")
    print(f"  With skill:    {total_time_yes:.2f}s, {total_tokens_yes} tokens")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_skill.py <python_file>")
        sys.exit(1)
    
    # Create output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_results_{timestamp}.log"
    
    # Redirect stdout to both console and file
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    with open(output_file, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, f)
        
        test_questions = [
            "What does this code do?",
            "Are there any bugs or issues?",
            "How can this code be improved?"
        ]
        
        benchmark(sys.argv[1], test_questions)
        
        sys.stdout = original_stdout
        print(f"\n✅ Results saved to: {output_file}")
