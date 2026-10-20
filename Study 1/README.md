# Abstract screening templates

Three notebooks, one response format each:

- `experiment_template_binary.ipynb` — include or exclude
- `experiment_template_yes_maybe_no.ipynb` — include, maybe, or exclude
- `experiment_template_5scale.ipynb` — relevance score from 1 to 5

In each notebook, set the dataset, criteria file, and example file in the configuration cell. The comments there list the runs that used that format. `RUN_EXPERIMENT` stays `False` until the printed prompt looks right.

## What you need

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put `OPENAI_API_KEY` in `.env`.

The screened abstracts are not in this folder. Point `DATA_PATH` at a local CSV with `title_full`, `abstract`, `stage_2`, and `stage_3`. See `data/README.md`. Criteria and few-shot examples are in `prompts/`. Results are written to `outputs/`.
