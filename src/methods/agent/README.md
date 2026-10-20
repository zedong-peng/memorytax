# Pi, Codex and Claude Code, one question at a time

Each question is a Harbor task with a fresh container and agent session. The container (`Dockerfile`) holds only `/workspace/conv-26/conv.json` (`data/playground/`, without questions or answers); the prompt is `prompt.txt`, the four rules of the Mem0 2025 answer template followed by the question. The agent's last message is its answer, judged with the Mem0 2025 judge like every other cell.

```sh
python src/methods/agent/run.py --agent pi --limit 2 --output results/smoke/pi          # print the configuration
python src/methods/agent/run.py --agent pi --limit 2 --output results/smoke/pi --run    # call the API
```

A full cell runs with `src/reproduce.py pi luna-high --run`. It needs Harbor 0.24.0, Docker and `uv`, and the model's and the judge's endpoints in `.env`.

- **Models**: GPT-4o-mini, and GPT-6 Luna and Claude Opus 5.5 at reasoning effort high. On the GPT models, Pi and Codex call the Responses API; Claude Code speaks only Anthropic Messages, which a local LiteLLM (1.104.2) translates. On Claude Opus 5.5, Pi and Claude Code call Anthropic and LiteLLM translates Codex's Responses API instead. The proxy listens only on the docker bridge, with a master key made fresh for each run.
- **Isolation**: while the agent runs, Harbor's egress control lets through only the model endpoint, or the proxy. Codex and Claude Code have web search off; Pi has none.
- **Versions**: Pi 1.1.0, Codex 0.159.2, Claude Code 2.1.295. Codex and Claude Code are installed in the image; Harbor installs Pi for each question. All use their default tools; on GPT-6 Luna, Codex picks its own code mode.
- **Limits**: 20 minutes a question; Pi and Claude Code at most 100 turns, Codex without a turn cap.
- **Failures**: a trial that raises or ends without an answer is deleted, and its question runs again in a new Harbor job, for up to four rounds. The accepted trial is kept in `trials/<question>/`: Harbor's result, the agent's raw log and its ATIF trajectory.
- **Cost**: a question costs the input and output tokens of every request in its trajectory, at list price, without cache discounts.
