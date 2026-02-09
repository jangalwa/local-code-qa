# Local Code Q&A Assistant

Privacy-first AI code assistant using small language models and agent skills. Runs entirely on local hardware with zero cloud dependencies.

📊 **[View Detailed Experiment Results](experiment.html)** - Can SLMs replace GPT-4? See the data.

## Features

- **Local execution** - No API keys, no data transmission
- **GPU accelerated** - MPS (Apple Silicon) and CUDA support
- **Agent skills** - Domain expertise injection via structured knowledge
- **RAG-based** - Semantic search with vector indexing
- **Benchmarking** - Quantifiable skill performance metrics

## Quick Start

```bash
git clone https://github.com/jangalwa/local-code-qa.git
cd local-code-qa
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/ai_project/file_qa.py your_code.py
```

## Architecture

```
Query → Embedding → Vector Search → Context Retrieval → LLM + Skill → Response
```

**Components:**
- LLM: Qwen2.5-3B-Instruct (3B params)
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Vector Index: PyTorch MPS-accelerated L2 search
- Skills: Structured domain knowledge (Python expert included)

## Usage

**Interactive Q&A:**
```bash
python src/ai_project/file_qa.py path/to/code.py
```

**Benchmark skills:**
```bash
python scripts/benchmark_skill.py path/to/code.py
```

## Performance

| Metric | Value |
|--------|-------|
| Response time | 30-40s |
| Memory usage | ~4GB |
| Skill overhead | +9% latency |
| Quality improvement | Significant (catches bugs base model misses) |

**Hardware tested:** MacBook Pro M2

## Project Structure

```
├── src/ai_project/
│   ├── file_qa.py          # Main Q&A system
│   ├── mps_index.py        # Vector search
│   └── conversation.py     # History management
├── skills/python-expert/   # Domain expertise
├── tests/                  # Test suite (30+ tests)
└── scripts/                # Benchmarking tools
```

## Agent Skills

Skills enhance model capabilities through structured knowledge injection. Format:

```markdown
---
name: skill-name
description: Brief description
---

# Skill Content
- Key concepts
- Code patterns
- Best practices
```

Place in `skills/skill-name/SKILL.md`

## Testing

```bash
pytest                              # All tests
pytest tests/test_mps_index.py     # Specific test
pytest --cov=src/ai_project        # With coverage
```

## Requirements

- Python 3.12+
- macOS (Apple Silicon) or CUDA GPU
- 8GB+ RAM

## Limitations

- Slower than cloud APIs (30-40s vs 2-5s)
- 32K token context window
- Best for: prototyping, learning, privacy-sensitive work
- Not suitable for: production systems, complex architecture decisions

## Technical Details

**RAG Pipeline:**
1. Token-aware document chunking with overlap
2. Semantic embedding generation
3. GPU-accelerated vector indexing
4. Top-k similarity search
5. Context-aware generation with skill injection

**Skill Impact (benchmarked):**
- Without skill: "No bugs found" (false negative)
- With skill: Identified 3 real issues with fixes
- Trade-off: 9% slower, significantly better quality

## License

MIT

## References

- HuggingFace "upskill" methodology
- Qwen model by Alibaba Cloud
- PyTorch, Transformers, sentence-transformers
