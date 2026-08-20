# NLP Coursework - Syntax, Classification & LLM Prompting

A 2 part NLP portfolio (Natural Language Processing). Each part is a
standalone Python script that prints its answers, with the captured output saved
alongside it.

## Declaration

“I have read and understood the sections of plagiarism in the College Policy on assessment offences and confirm that the work is my own, with the work of others
clearly acknowledged. I give my permission to submit my report to the plagiarism
testing database that the College is using and test it using plagiarism detection software, search engines or meta-searching software.”


## Parts

| Part | Script | Task |
|------|--------|------|
| **One** | `PartOne.py` | Syntax & style of 19th-century novels: type-token ratio, Flesch-Kincaid grade level, spaCy parsing, and direct-object / PMI analysis. |
| **Two** | `PartTwo.py` | Predicting a speaker's party from Hansard speeches with TF-IDF + LogisticRegression / ComplementNB, n-grams, and a custom spaCy tokenizer. |
| **Three** | `PartThree.py` | The same party-classification task using zero-shot and few-shot prompting with an open-weight LLM (Phi-4-mini via Ollama). |

## Data

All data lives in `texts/`:
- `novels/` — 12 plain-text novels (`Title-Author-Year.txt`).
- `hansard10000.csv` — speeches for Part Two.
- `hansard500.csv` — smaller sample for Part Three.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

NLTK data (`punkt`, `punkt_tab`, `cmudict`) is downloaded automatically the first
time `PartOne.py` runs.

**Part Three** additionally needs [Ollama](https://ollama.com/download) running,
with the model pulled:

```bash
ollama pull phi4-mini
```

## Running

```bash
python PartOne.py
python PartTwo.py
python PartThree.py
```

Captured runs are saved as `PartOne_output.txt`, `PartTwo_output.txt`, and
`PartThree_output.txt`. Text answers (Part Two tokenizer discussion, Part Three
model choice and comparison) are printed by the scripts and included in those files.

## Notes

- `.venv/` and `pickles/` (spaCy parse cache) are git-ignored and regenerated on run.
- Part Three uses in-context learning only — no model fine-tuning.