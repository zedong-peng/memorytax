# arXiv version

`main.tex` is `../index.html` as a static paper, with the same data. It adds what the page lacks: an abstract, the limits as a list, and the photo captions (the agents read the official LoCoMo file with its photo captions, the memory methods only the turn text), in Section 5 and Table 4.

| File | Role |
| --- | --- |
| `build.py` | `../data/*.json` and `../../data/locomo-conv26.json` → `figures/*.pdf`, `tables/*.tex`; prints the numbers the text quotes |
| `main.tex`, `refs.bib` | the paper |
| `main.bbl`, `main.pdf` | its build, committed so the tarball can be made without TeX |

Build on a machine with matplotlib (3.8.2 was used) and TeX Live, from this directory:

    python3 build.py
    latexmk -pdf main.tex

For arXiv, upload `arxiv.tar.gz` (pdflatex; the bibliography comes from `main.bbl`):

    tar czf arxiv.tar.gz main.tex main.bbl figures/*.pdf tables/*.tex
