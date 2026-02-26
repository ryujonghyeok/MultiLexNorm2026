# MultiLexNorm 2026 Demo
This repository provides a demo for the MultiLexNorm shared task.
It demonstrates how to download the dataset, run a simple baseline model (MFR), and evaluate normalization results.


- [**Full code is available here**](demo.ipynb).
- The datasets will be available at [development phase](https://huggingface.co/datasets/weerayut/multilexnorm2026-dev-pub) and [final phase](https://huggingface.co/datasets/weerayut/multilexnorm2026-dev-pub).
- Example MFR submission outputs: `outputs/submission_dev.zip` and `outputs/submission_full.zip`

## Set up the environment
```bash
# Create an environment and install packages
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Load data

```python
from datasets import load_dataset

pub_data = load_dataset("weerayut/multilexnorm2026-pub")

# Select a language
lang = "en"
en_train = pub_data["train"].filter(lambda x: x["lang"] == lang)
en_val = pub_data["validation"].filter(lambda x: x["lang"] == lang)
```


## Inference
```python
import pandas as pd
from utils import counting, mfr

# Smoke test the baseline
counts = counting(en_train)
mfr(['bcause', 'u', 'r', 'funny'], counts)

# Inference
ds = pd.DataFrame(en_val)
ds['pred'] = ds['raw'].apply(lambda x: mfr(x, counts))
```

## Evaluation
```python
from utils import evaluate

evaluate(
    raw=ds['raw'].tolist(),    # list[list[str]]
    gold=ds['norm'].tolist(),  # list[list[str]]
    pred=ds['pred'].tolist()   # list[list[str]]
)
```
Output:
```txt
Baseline acc.(LAI): 93.10
Accuracy:           97.37
ERR:                61.93
```
