# Uploading AutoValue AI to GitHub (step by step)

Git is installed (`git 2.55`), this folder is not a repo yet, and the `gh` CLI
is not installed — so the steps below use Git + the GitHub website only.

## Step 1 — Create a GitHub account (skip if you have one)

1. Go to https://github.com and sign up.
2. Sign in.

## Step 2 — Initialise the local repo and commit

Run these in PowerShell from `D:\carpriceprediction`:

```powershell
git init
git add .
git commit -m "AutoValue AI: car price prediction MVP"
git branch -M main
```

Check what will be committed first with `git status`. Expected files:
`PRD.md`, `README.md`, `GITHUB_UPLOAD.md`, `requirements.txt`, `.gitignore`,
`app.py`, `src/`, `data/cardekho_used_cars.csv`,
`data/DATA_DICTIONARY.md`, `notebooks/01_eda.ipynb`, `tests/`,
`models/metrics.json`, `models/car_price_pipeline.joblib` (~13 MB — fine,
GitHub's per-file limit is 100 MB).

## Step 3 — Create the empty repository on GitHub

1. On https://github.com, click **+ → New repository**.
2. Name it `autovalue-ai` (or `carpriceprediction`).
3. Set **Public** (or Private).
4. **Do NOT** tick "Add a README" (avoids merge conflicts).
5. Click **Create repository** and copy the HTTPS URL, e.g.
   `https://github.com/YOUR-USERNAME/autovalue-ai.git`.

## Step 4 — Link and push

```powershell
git remote add origin https://github.com/YOUR-USERNAME/autovalue-ai.git
git push -u origin main
```

If GitHub asks for credentials: use your username plus a
**Personal Access Token** (not your password) —
GitHub → Settings → Developer settings → Personal access tokens →
generate one with the `repo` scope. Save it somewhere safe.

## Step 5 — Verify

1. Refresh your repository page on GitHub — all files should be listed.
2. Confirm `README.md` renders on the repo home page.
3. Confirm `models/metrics.json` and the CSV are present.

## Step 6 — Polish the repo (recommended)

1. Click **About (gear icon)** → add description:
   "ML app estimating used-car prices (scikit-learn + Streamlit)" and topics:
   `machine-learning`, `streamlit`, `scikit-learn`, `regression`.
2. Add a screenshot: run `streamlit run app.py`, screenshot the app,
   save as `screenshots/app.png`, then `git add`, `git commit -m "Add app
   screenshot"`, `git push`.

## Later changes

```powershell
git add .
git commit -m "Describe your change"
git push
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `failed to push ... fetch first` | Someone (or the website) added files: run `git pull --rebase origin main`, then `git push`. |
| Password rejected | GitHub needs a Personal Access Token, not your password (see Step 4). |
| `remote origin already exists` | Run `git remote set-url origin <new-url>` instead of `add`. |
| Wrong files committed | Check `.gitignore`; unstage with `git reset <file>`. |
| Push is very slow | The model file is ~13 MB — first push takes a little while; later pushes are fast. |
