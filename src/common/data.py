"""Load conversation QA into one small representation."""
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Turn:
    id: str
    session: str
    date: str
    speaker: str
    text: str

    def format(self):
        return f"[Session {self.session} | {self.date}] {self.speaker}: {self.text}"


@dataclass
class Conversation:
    id: str
    turns: list[Turn]
    speaker_a: str | None = None

    def full_context(self):
        lines, previous = [], None
        for turn in self.turns:
            if turn.session != previous:
                lines.append(f"[{turn.date}]")
                previous = turn.session
            lines.append(f"{turn.speaker}: {turn.text}")
        return "\n".join(lines)


def load(source, dataset="locomo"):
    """Return (conversation, QA) pairs; IDs preserve duplicate questions."""
    if dataset != "locomo":
        raise ValueError("Only the official LoCoMo dataset is supported")
    rows = json.loads(source if isinstance(source, bytes) else Path(source).read_bytes())
    examples = []
    for item in rows:
        turns = []
        conversation = item["conversation"]
        sessions = sorted(int(m[1]) for key in conversation if (m := re.fullmatch(r"session_(\d+)", key)))
        for sid in sessions:
            date = conversation.get(f"session_{sid}_date_time", "")
            turns.extend(Turn(t.get("dia_id", ""), str(sid), date, t["speaker"], t["text"])
                         for t in conversation[f"session_{sid}"] if t.get("text", "").strip())
        sample = Conversation(str(item["sample_id"]), turns, conversation["speaker_a"])
        for i, qa in enumerate(item["qa"]):
            if qa["category"] not in (1, 2, 3, 4):
                continue
            answer = qa.get("answer", "") or qa.get("final_answer", "")
            if isinstance(answer, list):
                answer = answer[0] if answer else ""
            examples.append((sample, {
                "id": f"{sample.id}:{i}", "question": qa["question"], "reference": str(answer).strip(),
                "category": qa["category"], "evidence": qa.get("evidence", []),
            }))
    return examples
