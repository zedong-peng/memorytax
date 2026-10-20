# Mem0's 2025 evaluation code

`rag.py` is [mem0ai/mem0 `evaluation/src/rag.py`](https://github.com/mem0ai/mem0/blob/693e709389526b45cfadfd06d89a0e13af7c7345/evaluation/src/rag.py) (Apache-2.0, `LICENSE`) with one change: it loads `.env` when its class is built, not on import. Every method answers with its prompt and `generate_response`, through `src/common/answer.py`; the judge prompt that goes with it is in `src/common/mem0_judge.py`.
