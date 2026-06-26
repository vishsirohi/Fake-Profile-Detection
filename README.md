# Fake / Bot Twitter Profile Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/<your-repo>/blob/main/Fake_Profile_Detection_Colab.ipynb)

An end-to-end machine learning project that detects fake/bot Twitter accounts using both **account metadata** (behavioral features) and **tweet text** (NLP), combining classical ML, deep learning, and model explainability — all in a single, self-contained Colab notebook.

> Originally a classical ML project on basic Twitter account features. Rebuilt with a multi-model architecture (classical ML + deep learning + ensembling), proper evaluation methodology, SHAP explainability, and an interactive prediction demo.

## ▶️ Run it

Click the **Open in Colab** badge above, then `Runtime → Run all`. No setup required — the notebook installs its own dependencies and generates its own dataset. Takes ~2–5 minutes end-to-end on Colab's free CPU runtime.

## 🔍 What's inside

The single notebook (`Fake_Profile_Detection_Colab.ipynb`) walks through the full pipeline:

1. **Synthetic dataset generation** — 6,000 Twitter accounts with feature distributions modeled on patterns from published bot-detection research (Cresci et al. 2017; Kosmajac & Keselj 2019; Yang et al. 2020), with deliberate noise/overlap so the task isn't trivially separable
2. **Exploratory data analysis** — class balance, feature distributions, behavioral differences between bots and humans
3. **Feature engineering** — metadata features (follower ratios, posting cadence, profile completeness) and text features (TF-IDF, hand-crafted tweet statistics)
4. **Classical ML models** — Logistic Regression, Random Forest, XGBoost on account metadata
5. **NLP baseline** — TF-IDF + Logistic Regression on tweet text
6. **Deep learning** — a BiLSTM with learned embeddings, built and trained from scratch in PyTorch
7. **Stacked ensemble** — combines the metadata model + text model into a meta-classifier
8. **Explainability** — SHAP beeswarm, bar, and waterfall plots for the metadata model
9. **Live prediction demo** — a function you can call with any account's stats/tweets to get a bot-probability score

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (metadata) | 0.968 | 0.957 | 0.966 | 0.962 | 0.980 |
| Random Forest (metadata) | 0.973 | 0.966 | 0.970 | 0.968 | 0.982 |
| XGBoost (metadata) | 0.973 | 0.970 | 0.966 | 0.968 | 0.980 |
| TF-IDF + Logistic Regression (text) | 0.928 | 0.941 | 0.885 | 0.912 | 0.977 |
| BiLSTM (deep learning, text) | 0.922 | 0.931 | 0.879 | 0.904 | 0.971 |
| **Stacked Ensemble (metadata + text)** | **0.974** | **0.970** | **0.968** | **0.969** | **0.999** |

**Key finding:** the stacked ensemble achieves the best ROC-AUC (0.999) by combining a strong metadata model with a weaker-but-complementary text model — metadata catches behavioral red flags, text catches promotional language, and the ensemble recovers cases either model alone would miss.

## 🗂️ About the data

This project uses a **synthetically generated dataset** rather than a directly redistributed scrape of Twitter data. Real datasets like Cresci-2017 and TwiBot-20 require Twitter API access or research-only data-sharing agreements that aren't freely redistributable in a public repo. Instead, the notebook generates data whose feature distributions are explicitly grounded in published bot-detection research, with intentional class overlap and label noise so the problem isn't trivially solvable — a pitfall documented in some real benchmark datasets (parts of Cresci-2017 are near-perfectly separable by a single decision rule).

To use real data instead: swap the dataset-generation cell for `pd.read_csv('your_file.csv')` with the same column schema — the rest of the notebook works unchanged.

## 💬 Talking points for interviews

- *Why not just use accuracy?* Classes are imbalanced (~58/42); the notebook reports precision/recall/F1/ROC-AUC and uses balanced class weights.
- *Why ensemble two weaker text signals with a metadata model instead of just using the best single model?* The two models disagree on a meaningful subset of accounts (see the live demo's human-account example, where the text model alone says 92% bot but the ensemble correctly recovers "human" once metadata is factored in) — combining them improves overall ROC-AUC.
- *Why doesn't the BiLSTM beat the TF-IDF baseline?* A small from-scratch LSTM trained on a modest amount of short, casual text doesn't automatically outperform a well-tuned classical baseline — a realistic result rather than an inflated deep-learning win.
- *What would you do with more time/real data?* Swap in Cresci-2017/TwiBot-20 via API access, add graph-based features (follower network structure), try pretrained embeddings (BERT) instead of from-scratch LSTM embeddings.

## 📄 License

MIT — see [LICENSE](LICENSE).
