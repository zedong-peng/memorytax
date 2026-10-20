# Data

| File | Contents |
| --- | --- |
| `locomo-conv26.json` | LoCoMo's conversation conv-26, the entry of `locomo10.json` in [snap-research/locomo](https://github.com/snap-research/locomo) at `3eb6f2c`, unchanged: 19 sessions, 419 turns and 199 questions, of which the 152 in categories 1–4 are judged |
| `playground/conv-26/conv.json` | the agents' workspace: the same entry's `conversation` object written out unchanged, without questions or answers; byte for byte the file of the same name in [playground-locomo](https://github.com/zedong-peng/playground-locomo) at `37b2e73` |
| `manifest.json` | sources, commits and hashes, and how the conversation is cut from the official file |

LoCoMo is licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). To check the files:

```sh
shasum -a 256 -c - <<'EOF'
ef1eaea3e023cf376dda459a7dc355088dc9be05caf6bd8682a50a1db7898fc8  data/locomo-conv26.json
5827e7ec11c8ec8bf20cdbd380b0b59e0e31c58dfd1251bb01d42050e6b3a1a4  data/playground/conv-26/conv.json
EOF
```

To make them again from the official file:

```python
import json
rows = json.load(open("locomo10.json"))
conv = [r for r in rows if r["sample_id"] == "conv-26"]
open("data/locomo-conv26.json", "w").write(json.dumps(conv, indent=2))
open("data/playground/conv-26/conv.json", "w").write(json.dumps(conv[0]["conversation"], indent=2, ensure_ascii=False) + "\n")
```
