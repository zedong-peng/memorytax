"""BM25 over turns, each hit widened to its neighbouring turns."""
import math
import re
import string
from collections import Counter


PUNCTUATION = re.compile(f"[{re.escape(string.punctuation)}]")


def rank(texts, question):
    documents = [PUNCTUATION.sub(" ", text.lower()).split() for text in texts]
    if not documents:
        return []
    query = PUNCTUATION.sub(" ", question.lower()).split()
    df = Counter(word for tokens in documents for word in set(tokens))
    average_length = sum(map(len, documents)) / len(documents) or 1
    scores = []
    for i, tokens in enumerate(documents):
        counts = Counter(tokens)
        score = 0.0
        for word in query:
            if word in counts:
                idf = math.log(1 + (len(documents) - df[word] + 0.5) / (df[word] + 0.5))
                score += idf * counts[word] * 2.2 / (counts[word] + 1.2 * (0.25 + 0.75 * len(tokens) / average_length))
        scores.append((i, score))
    return [i for i, _ in sorted(scores, key=lambda pair: (-pair[1], pair[0]))]


def window_indices(turns, centers, width):
    """Turns within width of each center, inside the center's session, in first-seen order."""
    selected, seen = [], set()
    for center in centers:
        for i in range(max(0, center - width), min(len(turns), center + width + 1)):
            if turns[i].session == turns[center].session and i not in seen:
                seen.add(i)
                selected.append(i)
    return selected


def retrieve(sample, qa, top_k, window):
    """The top_k turns by BM25, each with its neighbours in the same session."""
    centers = rank([t.format() for t in sample.turns], qa["question"])[:top_k]
    selected = window_indices(sample.turns, centers, window)
    return {
        "context": "\n".join(sample.turns[i].format() for i in selected),
        "type": "turn_windows",
        "anchor_ids": [sample.turns[i].id for i in centers],
        "dialogue_ids": [sample.turns[i].id for i in selected],
    }
