# Paper data

`conv26_cumulative.json` holds, for each cell on LoCoMo conv-26, every question's category, the judge's verdict and its cost, and the cell's build cost. Costs are list prices from one table (`src/common/usage.py`), with every input token at the uncached rate and the judge left out; an agent's cost for a question is that of every request in its accepted trial. For Mem0 and Zep, `build_usd` is the build's cost as the run recorded it and `build_calls` the number of successful build calls.

The memory methods with GPT-4o-mini were run first, with the code this repository started from; their `rows` point into `results/legacy-gpt-4o-mini/`. Every other cell was run with this code: the memory methods by `src/reproduce.py`, the agents by `src/methods/agent/` (one Harbor container per question). `scripts/cells.py` copies them from `results/<method>/<model>/summary.json`. The run directories are not in the repository.

Time: a question's `seconds` is its wall time, end to end (in some Mem0 and Zep cells, retrieval and answer timed apart and added). A question with no time, interrupted and resumed, takes the cell's median; `seconds_imputed` counts them. `build_s` is the time at least one of conv-26's build calls was in flight: overlapping calls count once, pauses between resumed runs not at all. Times include each endpoint's speed and retries.

Usage: a question's `turns` and `tokens` are its model calls and their input plus output tokens, cached input included, embedding and judge calls left out; `build_turns` and `build_tokens` the same for the build.

`conv26_cross.json` holds the writer–reader cells: a Mem0 or Zep store written by one model and read by another, from `src/methods/<method>/reanswer.py`.
