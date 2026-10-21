# arXiv version

`main.tex` is `../index.html` as a static paper, with the same data. It adds what the page lacks: an abstract, the limits as a list, and the photo captions (the agents read the official LoCoMo file with its photo captions, the memory methods only the turn text), in Section 5 and Table 4.

| File | Role |
| --- | --- |
| `build.py` | `../data/*.json` and `../../data/locomo-conv26.json` → `tables/*.tex`, then `webfig.mjs` for the figures; prints the numbers the text quotes |
| `webfig.mjs` | opens `../index.html` in headless Chromium and writes each figure's own SVG to `figures/*.pdf`, one vector page per figure, so the paper draws what the page draws |
| `main.tex`, `refs.bib` | the paper |
| `main.bbl`, `main.pdf` | its build, committed so the tarball can be made without TeX |

Build on a machine with TeX Live, node 22 or later, and a Chromium (Playwright's, or `CHROME=/path/to/chrome`), from this directory:

    python3 build.py
    latexmk -pdf main.tex

For arXiv, upload `arxiv.tar.gz` (pdflatex; the bibliography comes from `main.bbl`):

    tar czf arxiv.tar.gz main.tex main.bbl figures/*.pdf tables/*.tex
