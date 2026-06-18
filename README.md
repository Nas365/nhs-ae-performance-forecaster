# NHS A&E Four-Hour Performance Forecaster

A lightweight web app that forecasts, one month ahead, how each major (Type 1) A&E department in England will perform against the NHS four-hour standard — the share of patients seen, admitted, transferred, or discharged within four hours of arrival.

---

## What it does

Select a hospital from the dropdown and see:

- **Recent trend** — the last 13 months of monthly performance for that department
- **Next-month forecast** — a data-driven estimate using a trained XGBoost model
- **Plain-language outlook** — how the forecast compares to the hospital's recent average, recent trend direction, and a note on the expected seasonal effect

Data is fetched live from NHS England on first load (cached for 24 hours) and falls back automatically to a bundled snapshot if the NHS England site is unavailable.

---

## The honest modelling result

On this short, recency-dominated tabular time series, **gradient-boosted trees (XGBoost) outperform an LSTM** — and both beat a persistence baseline (last month = next month):

| Model | Test MAE |
|---|---|
| Persistence baseline | 3.27 pp |
| **XGBoost — chosen model** | **2.93 pp** |
| Keras LSTM | 3.20 pp |

XGBoost is 10.4% better than persistence. Deep learning was tested rigorously under the same time-ordered evaluation and shown not to be the right tool here. The dominant signal is simply last month's performance (feature importance: 76%), not deep sequential patterns. Method choice was driven by evidence, not by what sounds advanced.

---

## Data source and licence

**Source:** NHS England, *A&E Attendances and Emergency Admissions*, monthly provider-level CSVs.  
Published at: <https://www.england.nhs.uk/statistics/statistical-work-areas/ae-waiting-times-and-activity/>

Contains public sector information licensed under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy to Streamlit Community Cloud

1. Fork this repository.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Select your fork, set the main file path to `app.py`, and deploy.

---

## Project structure

```
app.py                   Streamlit application
src/
  data.py                Live NHS data fetcher (find_csv_links, load_month, fetch_live_data)
  features.py            Feature builder for the XGBoost model
models/
  ae_model.joblib        Trained XGBoost model (retrained on all available months)
recent_history.csv       Bundled fallback: ~13 months of per-hospital history
notebooks/
  ae_project.ipynb       Full analysis: data acquisition, EDA, modelling, evaluation
requirements.txt
```

---

## Scope

- **Type 1 departments only** — major, 24/7, consultant-led A&Es. Type 3 (minor injury / walk-in) sees almost everyone within four hours and would mask real pressure.
- **Modelling window** — April 2023 onward (post-COVID stable regime; lag-12 feature requires 12 months of prior history).
- **Not for operational use** — forecasts carry roughly ±3 percentage points of uncertainty and are illustrative only.
