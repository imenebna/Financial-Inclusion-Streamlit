# Financial Inclusion: Streamlit App (Checkpoint)

This repository contains:
- `Financial_Inclusion_Checkpoint.ipynb` (training + evaluation notebook)
- `app.py` (Streamlit app for inference)
- `artifacts/` (exported model + metadata used by the app)

## 1) Generate artifacts from the notebook
1. Open `Financial_Inclusion_Checkpoint.ipynb` in Colab/Jupyter.
2. Run all cells.
3. Ensure the export cell creates:
   - `artifacts/model.joblib`
   - `artifacts/metadata.json`

These two files are required for the Streamlit app.

## 2) Run the app locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 3) Deploy on Streamlit Community Cloud
1. Push this repository to GitHub (include `app.py`, `requirements.txt`, and the `artifacts/` folder).
2. On Streamlit Community Cloud, create a new app from the GitHub repo.
3. Set:
   - Main file: `app.py`

## Notes for evaluation
- The app builds input fields dynamically from `metadata.json` (categorical selectboxes + numeric number inputs).
- A **Validate & Predict** button triggers prediction and displays both the predicted label and probability.
