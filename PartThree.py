import ollama
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

MODEL = "phi4-mini"
PARTIES = ["Conservative", "Labour", "Scottish National Party", "Liberal Democrat"]



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

