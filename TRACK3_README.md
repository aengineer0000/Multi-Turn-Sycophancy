# Track 3: Behavioral Baselines & Confounder Auditing

This folder implements Track 3 of the Multi-Turn Sycophancy project.

## Goal

Track 3 answers two questions:

1. **Behavioral baseline:** When the target model receives only Turn 1 (correct tutor statement) and Turn 2 (student pushback), how often does it naturally capitulate on Turn 3?
2. **Confounder audit:** Are the synthetic firm vs. sycophantic completions trivially distinguishable from superficial cues such as token length or repeated phrases?

## Files

- `src/eval/behavioral_baseline.py` — generates natural Turn 3 responses from the target model.
- `src/eval/score_behavior.py` — labels each generated response `CAVED`, `FIRM`, or `UNCERTAIN`; supports a Gemini semantic judge plus a transparent lexical fallback.
- `src/eval/audit_confounders.py` — token-length audit, length-only classifier, TF-IDF and 1–3 gram audit.
- `scripts/run_track3.sh` — convenience runner for smoke and full experiments.
- `requirements-track3.txt` — Python dependencies for Track 3.

## Experimental rule

The behavioral baseline **must not** show the target model `final_turn_sycophantic` or `final_turn_firm`. It only uses `context_turns` and asks the model to generate the next Tutor message.

The prompt is deliberately neutral. Do not add instructions such as "never agree with the student" or "maintain factual accuracy," because that would steer the model and invalidate the natural baseline.

## Recommended target model

Use the **same exact checkpoint as Track 1/Track 2**. If the team has not fixed one yet, `google/gemma-2-2b-it` is the default here because it is an instruction-tuned open-weight model suitable for tutor dialogue. If the team chooses another model, pass it with `--model` or set `MODEL_NAME`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-track3.txt
```

If Gemma access is gated on Hugging Face, accept the model license on Hugging Face and then run:

```bash
hf auth login
```

Optional semantic judge:

```bash
cp .env.example .env
```

Then edit `.env` and set `GOOGLE_API_KEY`. Never commit `.env`.

## First run: smoke test

```bash
bash scripts/run_track3.sh smoke
```

This generates only five natural Turn 3 responses. Manually inspect:

```text
data/processed/behavioral_results.json
```

Verify that:

- only Turn 1 + Turn 2 are in each prompt,
- the stored synthetic Turn 3 answers are not present,
- the model generated a clean Tutor reply,
- responses are not obviously truncated.

## Full run

After the five-sample smoke test is correct:

```bash
bash scripts/run_track3.sh full
```

If `GOOGLE_API_KEY` is configured, the full runner uses the semantic judge. Otherwise it uses the lexical fallback and places ambiguous cases in `reports/tables/manual_review.csv`.

## Key outputs

### Behavioral baseline

- `data/processed/behavioral_results.json`
- `data/processed/behavioral_scored.csv`
- `reports/tables/behavioral_summary.json`
- `reports/tables/capitulation_by_topic.csv`
- `reports/figures/capitulation_rate.png`
- `reports/figures/capitulation_by_topic.png`

Natural capitulation rate is reported as:

```text
CAVED / (CAVED + FIRM)
```

`UNCERTAIN` samples are reported separately rather than silently forcing them into either class.

### Confounder audit

- `reports/tables/confounder_summary.csv`
- `reports/tables/confounder_audit_results.json`
- `reports/tables/lexical_artifacts.csv`
- `reports/figures/token_length_distribution.png`
- `reports/figures/token_length_boxplot.png`
- `reports/figures/top_lexical_artifacts.png`

The length-only logistic-regression baseline is especially important. If it approaches the activation probe's performance, response length may be a major confounder.

## Reproducibility

The default random seed is `42`. Generation defaults match the project outline:

- temperature: `0.7`
- max new tokens: `256`

Record the exact target model checkpoint used in the final report.
