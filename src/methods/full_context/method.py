"""Full context: the whole conversation with its session dates, the upper bound."""

from common.answer import document


def retrieve(sample, qa):
    return {"context": document(sample), "type": "conversation", "dialogue_ids": [t.id for t in sample.turns]}
