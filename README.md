# MemoryTax: The Bitter Lesson of Agent Memory

Code and data for [MemoryTax](https://zedongpeng.com/memorytax/) ([arXiv:2610.29823](https://arxiv.org/abs/2610.29823)). On one LoCoMo conversation (conv-26, 152 judged questions) it runs four memory methods (full context, BM25, Mem0, Zep) and three coding agents that search the raw conversation file (Pi, Codex, Claude Code), each with GPT-4o-mini, GPT-6 Luna and Claude Opus 5.5, and records every answer, verdict and cost.

```text
data/      LoCoMo conv-26 and the agents' workspace (data/README.md)
papers/    the paper: index.html, its data and scripts, and the arXiv version (papers/README.md)
src/       the runners; src/reproduce.py is the entry point (src/README.md)
results/   run outputs, not in git: results/<method>/<model>/
```

## Running

```sh
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_sm
cp .env.example .env                                # a key and an endpoint for each model
.venv/bin/python src/check_api.py                   # a few real requests: keys, endpoints, usage fields
.venv/bin/python src/reproduce.py                   # every cell and its state
.venv/bin/python src/reproduce.py bm25 opus --run   # run one cell; the same command resumes
```

Zep runs in its own environment, `.venv-zep`, from `src/methods/zep/requirements.txt`. The agents need Harbor 0.24.0, Docker and `uv` (`src/methods/agent/README.md`).

Costs are list prices, whatever an endpoint charges. Our runs reached GPT-4o-mini (also the judge), the embeddings and GPT-6 Luna through OpenAI-compatible routers, and Claude Opus 5.5 through Anthropic's API.

## Licenses

The code is MIT (`LICENSE`). The LoCoMo data is CC BY-NC 4.0 (`src/common/locomo/LICENSE`). Mem0's evaluation code and prompts are Apache-2.0 (`src/common/mem0_eval/LICENSE`, `src/methods/mem0/LICENSE`).

## Citation

```bibtex
@misc{peng2026memorytax,
  title         = {{MemoryTax: The Bitter Lesson of Agent Memory}},
  author        = {Peng, Zedong},
  year          = {2026},
  eprint        = {2610.29823},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.29823},
}
```
