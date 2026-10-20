# The container for one question: conv-26/conv.json in /workspace and nothing else of the dataset.
# Codex and Claude Code are installed at the versions run.py pins, so Harbor skips installing them for
# every question; Pi is installed by Harbor (through nvm) each time.
FROM ubuntu:24.04
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl procps python3 ripgrep xz-utils \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://nodejs.org/dist/v24.21.0/node-v24.21.0-linux-x64.tar.xz \
    | tar -xJ -C /usr/local --strip-components=1 \
    && npm install -g @openai/codex@0.159.2 \
    && curl -fsSL https://downloads.claude.ai/claude-code-releases/bootstrap.sh | bash -s -- 2.1.295
ENV PATH=/root/.local/bin:$PATH
COPY conv-26/conv.json /workspace/conv-26/conv.json
WORKDIR /workspace
