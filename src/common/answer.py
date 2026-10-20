"""The answer step every method shares: Mem0's 2025 evaluation prompt, run by its own code.

The prompt and generate_response come from mem0ai/mem0 evaluation/src/rag.py (common/mem0_eval/rag.py,
Apache-2.0); only the API call goes through the metered client.
"""

from types import SimpleNamespace

from common.mem0_eval.rag import RAGManager


def document(sample):
    return "".join(f"{t.date} | {t.speaker}: {t.text}\n" for t in sample.turns)


def generate_answer(client, question, context):
    """Execute upstream's answer function through the metered API client."""

    def create(*, model, messages, temperature):
        content = client.chat(
            messages[0]["content"],
            messages[1]["content"],
            max_tokens=None,
            seed=None,
            role="answer",
            transport_attempts=1,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    manager = RAGManager.__new__(RAGManager)
    manager.model = client.connections["answer"]["model"]
    manager.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return manager.generate_response(question, context)[0]
