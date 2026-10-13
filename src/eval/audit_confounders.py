#!/usr/bin/env python3
"""Audit Track 3 confounders in the synthetic contrastive dataset.

Checks:
1. Token-length distributions for synthetic sycophantic vs firm Turn 3s.
2. Length-only logistic-regression baseline.
3. TF-IDF / 1-3 gram lexical artifacts.
4. Simple statistical summaries and Mann-Whitney U test.

Outputs are written to reports/tables/ and reports/figures/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer


DEFAULT_DATA = "data/raw/cs_tutor_dataset.json"
DEFAULT_MODEL = "google/gemma-2-2b-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Track 3 dataset confounders")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Tokenizer used for exact token lengths")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_dialogues(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("dialogues", "data", "samples"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError("Could not find dialogue list in dataset")


def completion_text(dialogue: dict[str, Any], key: str) -> str:
    value = dialogue.get(key, "")
    if isinstance(value, dict):
        return str(value.get("text", "")).strip()
    return str(value).strip()


def build_dataframe(dialogues: list[dict[str, Any]], tokenizer: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dialogue_id, dialogue in enumerate(dialogues):
        for label, key in (
            ("sycophantic", "final_turn_sycophantic"),
            ("firm", "final_turn_firm"),
        ):
            text = completion_text(dialogue, key)
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            rows.append(
                {
                    "dialogue_id": dialogue_id,
                    "topic": dialogue.get("topic", ""),
                    "label": label,
                    "text": text,
                    "token_length": len(token_ids),
                    "character_length": len(text),
                    "word_length": len(text.split()),
                }
            )
    return pd.DataFrame(rows)


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    pooled_var = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    if pooled_var <= 0:
        return 0.0
    return float((x.mean() - y.mean()) / np.sqrt(pooled_var))


def save_length_analysis(df: pd.DataFrame) -> dict[str, Any]:
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    stats = (
        df.groupby("label")["token_length"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    stats.to_csv("reports/tables/confounder_summary.csv", index=False)

    syc = df.loc[df["label"] == "sycophantic", "token_length"].to_numpy()
    firm = df.loc[df["label"] == "firm", "token_length"].to_numpy()

    u_stat, p_value = mannwhitneyu(syc, firm, alternative="two-sided")
    d = cohen_d(syc, firm)

    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(syc, bins=25, alpha=0.55, label="Sycophantic")
    plt.hist(firm, bins=25, alpha=0.55, label="Firm")
    plt.xlabel("Token length")
    plt.ylabel("Number of responses")
    plt.title("Token-Length Distribution: Sycophantic vs Firm")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/token_length_distribution.png", dpi=300)
    plt.close()

    # Boxplot
    plt.figure(figsize=(6.5, 4.5))
    plt.boxplot([syc, firm], labels=["Sycophantic", "Firm"])
    plt.ylabel("Token length")
    plt.title("Token-Length Sanity Check")
    plt.tight_layout()
    plt.savefig("reports/figures/token_length_boxplot.png", dpi=300)
    plt.close()

    return {
        "mann_whitney_u": float(u_stat),
        "mann_whitney_p": float(p_value),
        "cohen_d_syc_minus_firm": float(d),
    }


def length_only_baseline(df: pd.DataFrame, seed: int) -> dict[str, Any]:
    """Can token length alone predict the class?

    We split by dialogue_id so the two members of a contrastive pair never
    leak across train and test sets.
    """
    X = df[["token_length"]].to_numpy(dtype=float)
    y = (df["label"] == "sycophantic").astype(int).to_numpy()
    groups = df["dialogue_id"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X[train_idx], y[train_idx])

    pred = clf.predict(X[test_idx])
    prob = clf.predict_proba(X[test_idx])[:, 1]

    return {
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "accuracy": float(accuracy_score(y[test_idx], pred)),
        "roc_auc": float(roc_auc_score(y[test_idx], prob)),
        "coefficient_for_token_length": float(clf.coef_[0][0]),
        "intercept": float(clf.intercept_[0]),
    }


def lexical_audit(df: pd.DataFrame) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 3),
        min_df=2,
        max_features=10000,
        strip_accents="unicode",
    )
    X = vectorizer.fit_transform(df["text"])
    features = vectorizer.get_feature_names_out()

    syc_mask = (df["label"].to_numpy() == "sycophantic")
    firm_mask = (df["label"].to_numpy() == "firm")

    syc_mean = np.asarray(X[syc_mask].mean(axis=0)).ravel()
    firm_mean = np.asarray(X[firm_mask].mean(axis=0)).ravel()
    diff = syc_mean - firm_mean

    lex = pd.DataFrame(
        {
            "ngram": features,
            "sycophantic_mean_tfidf": syc_mean,
            "firm_mean_tfidf": firm_mean,
            "syc_minus_firm": diff,
            "absolute_difference": np.abs(diff),
        }
    ).sort_values("absolute_difference", ascending=False)

    lex.to_csv("reports/tables/lexical_artifacts.csv", index=False)

    top_syc = lex.sort_values("syc_minus_firm", ascending=False).head(15)
    top_firm = lex.sort_values("syc_minus_firm", ascending=True).head(15)

    plot_df = pd.concat(
        [
            top_syc.assign(direction="Sycophantic-associated"),
            top_firm.assign(direction="Firm-associated"),
        ]
    )
    plot_df["plot_value"] = plot_df["syc_minus_firm"]
    plot_df = plot_df.sort_values("plot_value")

    plt.figure(figsize=(9, 9))
    plt.barh(plot_df["ngram"], plot_df["plot_value"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Mean TF-IDF difference (sycophantic - firm)")
    plt.title("Most Discriminative Lexical n-grams")
    plt.tight_layout()
    plt.savefig("reports/figures/top_lexical_artifacts.png", dpi=300)
    plt.close()

    return lex


def main() -> None:
    args = parse_args()
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    dialogues = load_dialogues(args.data)
    print(f"Loaded {len(dialogues)} dialogues")
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    df = build_dataframe(dialogues, tokenizer)
    df.to_csv("reports/tables/confounder_dataset.csv", index=False)

    statistical_results = save_length_analysis(df)
    length_baseline = length_only_baseline(df, seed=args.seed)
    lexical_audit(df)

    result = {
        "dialogues": int(len(dialogues)),
        "contrastive_responses": int(len(df)),
        "target_tokenizer": args.model,
        "length_statistics_test": statistical_results,
        "length_only_classifier": length_baseline,
    }

    with open("reports/tables/confounder_audit_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nConfounder audit complete")
    print(json.dumps(result, indent=2))
    print("\nKey files:")
    print("  reports/figures/token_length_distribution.png")
    print("  reports/figures/top_lexical_artifacts.png")
    print("  reports/tables/confounder_summary.csv")
    print("  reports/tables/lexical_artifacts.csv")


if __name__ == "__main__":
    main()
