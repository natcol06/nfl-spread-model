"""Train, evaluate, and chart the NFL win and spread models."""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             log_loss, confusion_matrix)

from data import load_raw
from features import build_features, FEATURES

TRAIN_END = 2023
FIG_DIR = "../figures"


def split(model_df, train_end=TRAIN_END):
    train = model_df[model_df["season"] <= train_end]
    test = model_df[model_df["season"] > train_end]
    return train, test


def fit_model(train, target):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipe.fit(train[FEATURES], train[target])
    return pipe


def evaluate(pipe, test, target, label):
    probs = pipe.predict_proba(test[FEATURES])[:, 1]
    preds = (probs > 0.5).astype(int)

    print(f"\n{label}  (n={len(test)})")
    print("  accuracy:", round(accuracy_score(test[target], preds), 3))
    print("  auc:     ", round(roc_auc_score(test[target], probs), 3))
    print("  log loss:", round(log_loss(test[target], probs), 3))
    return probs, preds


def plot_coefficients(pipe, path):
    coefs = pipe.named_steps["logisticregression"].coef_[0]
    cdf = pd.DataFrame({"feature": FEATURES, "coefficient": coefs})
    cdf = cdf.sort_values("coefficient")

    plt.figure(figsize=(7, 4))
    sns.barplot(data=cdf, x="coefficient", y="feature",
                hue="feature", legend=False)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("What drives the win prediction")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion(test, preds, target, path):
    cm = confusion_matrix(test[target], preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["pred loss", "pred win"],
                yticklabels=["actual loss", "actual win"])
    plt.title("Home team win predictions")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    pbp, games = load_raw()
    model_df = build_features(pbp, games)
    train, test = split(model_df)
    print(f"train {len(train)} games, test {len(test)} games")

    win_model = fit_model(train, "home_won")
    _, win_preds = evaluate(win_model, test, "home_won", "Straight up")

    ats_model = fit_model(train, "home_covered")
    evaluate(ats_model, test, "home_covered", "Against the spread")
    print("  breakeven needed: 0.524")

    plot_coefficients(win_model, f"{FIG_DIR}/coefficients.png")
    plot_confusion(test, win_preds, "home_won", f"{FIG_DIR}/confusion.png")
    print(f"\nCharts written to {FIG_DIR}/")


if __name__ == "__main__":
    main()