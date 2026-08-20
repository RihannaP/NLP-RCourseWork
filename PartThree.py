import ollama
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

MODEL = "phi4-mini"
PARTIES = ["Conservative", "Labour", "Scottish National Party", "Liberal Democrat"]

def load_and_clean(path=Path.cwd() / "texts" / "hansard500.csv"):
    # same cleaning as part2 but restricted to the SAME four party labels
    # as part2 rather than recomputing 'four most common' on this smaller file
    # (whose 4th commonest is actually the DUP, not the Liberal Democrats)
    df = pd.read_csv(path)
    df["party"] = df["party"].replace("Labour (Co-op)", "Labour")
    df = df[df["party"].isin(PARTIES)]          # same label set as part2
    df = df[df["speech_class"] == "Speech"]
    df = df[df["speech"].str.len() >= 1000]
    return df


def get_split(df):
    #80/20 split with seed 26 (same test_size and seed as part2). I cannot
    # stratify here because Liberal Democrat has only one qualifying speech in the
    # 500-row sample, so a plain random split is used.
    train, test = train_test_split(df, test_size=0.2, random_state=26)
    return train, test

ZERO_SHOT_SYSTEM = (
    "You are a political analyst. You are given an excerpt from a UK parliamentary "
    "speech. Classify which party the speaker belongs to. "
    "Answer with EXACTLY ONE of these labels and nothing else:\n"
    "Conservative\nLabour\nScottish National Party\nLiberal Democrat"
)

def parse_label(raw):
    #map the model's free-text output to one of the four labels
    text = raw.strip().lower()
    if "scottish national" in text or "snp" in text:
        return "Scottish National Party"
    if "liberal democrat" in text or "lib dem" in text:
        return "Liberal Democrat"
    if "labour" in text:
        return "Labour"
    if "conservative" in text or "tory" in text:
        return "Conservative"
    return "Conservative"   # fallback if output is unrecognisable (majority class)

def classify_zero_shot(speech):
    #send one speech to the model and return its predicted party label
    resp = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": ZERO_SHOT_SYSTEM},
            {"role": "user", "content": f"Speech:\n{speech}\n\nParty:"},
        ],
    #num_predict limits generated tokens as I only need a short label so this makes it faster
        options={"temperature": 0, "num_predict": 10}, 
    )
    raw = resp["message"]["content"]
    print("RAW ->", repr(raw))       # see exactly what the model says
    return parse_label(raw)


def evaluate(test, predict_fn, label):
    #run a prediction function over the test set and print metrics
    preds = [predict_fn(s) for s in test["speech"]]
    y_true = test["party"].tolist()
    print(f"\n===== {label} =====")
    print("macro-F1:", f1_score(y_true, preds, average="macro", zero_division=0))
    print(classification_report(y_true, preds, zero_division=0))
    return preds


ANSWER_3A = """
========================================
3(a) - Model, access route, and generation parameters
========================================
Model:        Phi-4-mini-instruct (Ollama tag 'phi4-mini'), a 3.8B-parameter
              open-weight, instruction-tuned model from Microsoft, running at
              Q4_K_M quantisation.
Access route: Ollama, running locally on an Apple M5 Pro (Metal-accelerated).
Parameters:   temperature = 0 (deterministic / greedy decoding).

Why these choices:
- Instruction-tuned model: as covered in the Week 7 lecture, a base model only
  performs text completion and will not reliably follow a task instruction, so a
  party-classification task requires an instruction-tuned model. Phi-4-mini-instruct
  is the model we used in the Week 6 lab (query-test.py).
- Open-weight and local: Phi-4-mini is open-weight and small enough to run locally,
  following the Lab 6 guidance to match model size to available hardware. Running it
  locally through Ollama also means no speech data leaves the machine.
- Ollama route: I first tried the HuggingFace transformers pipeline (the Lab 6
  approach), but the torch/accelerate stack segfaulted on this machine's Python 3.14
  while loading the weights. Ollama runs the same model through its own Metal engine,
  avoiding that stack entirely. The brief explicitly permits Ollama.
- temperature = 0: classification should be deterministic and reproducible, so I use
  greedy decoding rather than sampling. This removes run-to-run variation and makes
  the reported macro-F1 stable.
"""

if __name__ == "__main__":
    print(ANSWER_3A)

    df = load_and_clean()
    print("cleaned shape:", df.shape)
    train, test = get_split(df)
    print("train/test sizes:", len(train), len(test))

    print("\n########## 3(b): ZERO-SHOT ##########")
    print("---- exact zero-shot prompt (system message) ----")
    print(ZERO_SHOT_SYSTEM)
    evaluate(test, classify_zero_shot, "Zero-shot (phi4-mini)")