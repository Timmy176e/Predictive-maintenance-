# Predictive Maintenance — Streamlit App

A Streamlit UI for the Random Forest failure-risk model trained on the
AI4I 2020 Predictive Maintenance dataset (from your notebook).

## Files

```
.
├── app.py                 # the Streamlit app
├── train_model.py         # reproduces the notebook's final RF pipeline and saves it
├── requirements.txt
├── .gitignore
└── model_artifacts/
    ├── random_forest_pipeline.joblib   # you need to add this (see below)
    └── model_metadata.json             # you need to add this (see below)
```

## 1. Get the trained model into `model_artifacts/`

The app loads a saved model from `model_artifacts/`. You have two options:

**Option A — reuse what you already trained.** The last cell of your
notebook already copied the artifacts to Google Drive, at
`MyDrive/Industrial_Maintenance_AI/model_artifacts/`. Download
`random_forest_pipeline.joblib` and `model_metadata.json` from there and
place both files into this project's `model_artifacts/` folder.

**Option B — retrain from scratch.** Run, wherever you have normal internet
access (Colab or your own machine):

```bash
pip install scikit-learn pandas joblib
python train_model.py
```

This reproduces the exact preprocessing + `RandomForestClassifier` pipeline
from your notebook (same hyperparameters, same 80/20 split, `random_state=42`)
and writes both files into `model_artifacts/` for you.

> I couldn't run this for you from here — `archive.ics.uci.edu`, where the
> dataset lives, isn't reachable from this sandboxed environment — so one of
> the two options above is needed before the app has a real model to load.

## 2. Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Enter Type, air/process temperature,
rotational speed, torque and tool wear, and it returns a failure probability
and a risk call against the saved decision threshold.

## 3. Deploy on Streamlit Community Cloud (free)

1. Push this folder to a **public** (or private, on a paid plan) GitHub repo
   — `model_artifacts/*.joblib` included. A 400-tree Random Forest on this
   dataset is typically a few MB, well under GitHub's 100 MB file limit.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. **New app** → pick the repo/branch → set **Main file path** to `app.py`
   → **Deploy**.
4. Streamlit Cloud installs `requirements.txt` and starts the app; you get a
   public `*.streamlit.app` URL to share.

If you update the model later, just overwrite the two files in
`model_artifacts/` and push — Streamlit Cloud redeploys on every push to the
branch it's tracking.

## Notes

- `app.py` reads the trained pipeline's own column names and `Type`
  categories at load time rather than hard-coding them, so it keeps working
  even if your column headers differ slightly (e.g. unit formatting).
- If you see a scikit-learn version-mismatch warning when the app loads the
  model, pin `scikit-learn` in `requirements.txt` to the exact version
  recorded in `model_artifacts/model_metadata.json` under
  `"scikit_learn_version"`.
- The "About this model" panel in the app surfaces the precision/recall/
  ROC-AUC and limitations already captured in `model_metadata.json`.
