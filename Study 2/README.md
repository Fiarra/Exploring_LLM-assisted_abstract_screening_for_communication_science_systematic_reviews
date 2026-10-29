# Automated Abstract Screening

Classify abstracts against inclusion and exclusion criteria using an LLM, via an in-house university API or the OpenAI client.

## Folder structure

| Folder | Content |
|---|---|
| `prompts/` | System and user prompts. The system prompt is fixed; the user prompt contains a placeholder for the decision type (`binary`, `ordinal`, `likert`). |
| `criteria/` | Inclusion and exclusion criteria as `.csv` with the columns `criteria` (e.g. `IC1`) and `type` (`inclusion` / `exclusion`). |
| `examples/` | Example entries (`.csv`) that can be passed to the model as few-shot examples. |
| `results/` | `result.csv` and log files with metadata. All results are written here. |
| `utils/` | Core code: `decisions` (handling of the decision types `binary`, `ordinal`, `likert`) and `main` (the screening logic). |

## Setup

### 1. Provide your data

At the moment, the data is stored on an external cloud. Create your own `settings.py` based on `settings.default.py` so the code can load your data.

### 2. Configure the API

You can choose which API to use. Here, the in-house API of the university server is used. Both plain URL requests and the OpenAI client are supported.

Create a `.env` file and set `API_URL` and `API_KEY`. They are loaded automatically.

> **Note:** Never commit your API key. Add `.env` to `.gitignore` and share only a `.env.example` with placeholders.

## Test case

`test.py` is an executable test case, once API key is provided. Minimal example:

```python
res = classify_one_entry(
    project_id="climate",
    entry={"id": "A1", "abstract": "We analyze media framing of protests climate change..."},
    system_prompt_path="prompts/system.txt",
    user_prompt_path="prompts/user.txt",
    model="mistral-small",
    decision="binary",                  # binary | ordinal | likert
    temperature=0.0,
    topic="climate communication",
    domain="communication research",
    codebook_csv="criteria/criteria_test.csv",
    examples_csv="examples/example_test.csv",
)

res
```
