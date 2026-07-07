"""
WASH -> Incidence Forecast Dashboard (simple version)
------------------------------------------------------
Run with:  streamlit run dashboard_app.py

Expects these files in the same folder:
  - MERGED_normalized.csv
  - tuned_results.json

Trains one CatBoost model at startup (cached) using the exact features,
temporal split, and hyperparameters from your Model_Training.ipynb, then
lets a user pick a country and see history + next-year prediction versus
the naive baselines.
"""

import json

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor

SEED = 42
DATA_PATH = "MERGED_normalized.csv"
PARAMS_PATH = "tuned_results.json"
MODEL_KEY = "CatBoost (Normalized)"  # best-performing model from your comparison table

st.set_page_config(page_title="WASH Incidence Forecast", layout="wide")


# ---------- data + model (cached so this only runs once) ----------

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Incidence_next_log"] = np.log1p(df["Incidence_next"])
    return df


@st.cache_resource
def train_model(df):
    model_df = pd.get_dummies(df, columns=["WHO_Region"], dummy_na=True)
    exclude_cols = ["Country", "Year", "Autochthonous_Cases", "Population",
                     "Incidence", "Incidence_next", "Incidence_next_log"]
    feature_cols = [c for c in model_df.columns if c not in exclude_cols]

    X = model_df[feature_cols]
    y_log = model_df["Incidence_next_log"]
    y_orig = df["Incidence_next"]
    years = df["Year"]

    train_mask = years <= 2019
    test_mask = years > 2019

    with open(PARAMS_PATH) as f:
        tuned = json.load(f)
    params = tuned[MODEL_KEY]["best_params"]

    model = CatBoostRegressor(random_state=SEED, verbose=0,
                               bootstrap_type="Bernoulli", **params)
    model.fit(X[train_mask], y_log[train_mask])

    # temporal test performance, for the baseline-comparison panel
    pred_log = model.predict(X[test_mask])
    pred = np.clip(np.expm1(pred_log), a_min=0, a_max=None)
    rmse = float(np.sqrt(np.mean((y_orig[test_mask].values - pred) ** 2)))
    mean_pred_rmse = float(np.sqrt(np.mean(
        (y_orig[test_mask].values - y_orig[train_mask].mean()) ** 2)))

    return model, feature_cols, model_df, rmse, mean_pred_rmse


df = load_data()
model, feature_cols, model_df, model_rmse, mean_rmse = train_model(df)


# ---------- sidebar controls ----------

st.sidebar.title("Controls")
countries = sorted(df["Country"].dropna().unique())
country = st.sidebar.selectbox("Country", countries, index=countries.index("Afghanistan")
                                if "Afghanistan" in countries else 0)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Model:** CatBoost (Normalized features)\n\n"
    f"**Temporal test RMSE:** {model_rmse:.1f}\n\n"
    f"**Naive mean-predict RMSE:** {mean_rmse:.1f}\n\n"
    "This model is only marginally better than a naive baseline — "
    "treat predictions as a rough trend signal, not a precise forecast."
)


# ---------- main panel ----------

st.title("WASH Indicators -> Disease Incidence Forecast")
st.caption("Country-year model trained on JMP water/sanitation/hygiene access "
           "indicators to predict next-year incidence per 100,000 population.")

country_df = df[df["Country"] == country].sort_values("Year")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Incidence history — {country}")
    st.line_chart(country_df.set_index("Year")["Incidence"])

with col2:
    latest_idx = country_df.index[-1]
    latest_year = int(country_df.loc[latest_idx, "Year"])
    latest_incidence = country_df.loc[latest_idx, "Incidence"]

    row = model_df.loc[[latest_idx], feature_cols]
    pred_log = model.predict(row)[0]
    pred = float(np.clip(np.expm1(pred_log), a_min=0, a_max=None))

    st.subheader(f"Forecast for {latest_year + 1}")
    st.metric("Predicted incidence", f"{pred:.1f}",
              delta=f"{pred - latest_incidence:+.1f} vs {latest_year}"
              if pd.notna(latest_incidence) else None)

    st.caption("Baselines for context:")
    st.write(f"- Persistence (repeat {latest_year} value): "
             f"{latest_incidence:.1f}" if pd.notna(latest_incidence) else
             "- Persistence: unavailable (missing current-year incidence)")
    st.write(f"- Global mean-predict baseline: {df['Incidence_next'].mean():.1f}")

st.markdown("---")
st.subheader("Underlying WASH indicators (most recent year)")
display_cols = ["Year"] + [c for c in ["wat_basal_t", "san_basal_t", "hyg_bas_t"] if c in country_df.columns]
st.dataframe(country_df[display_cols].tail(5).set_index("Year"))
