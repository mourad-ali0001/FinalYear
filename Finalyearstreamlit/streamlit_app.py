import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path

st.set_page_config(page_title="EPL Outcome Prediction Dashboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parent

CV_CSV = BASE_DIR / "cv_results.csv"
TEST_CSV = BASE_DIR / "test_results.csv"
BAL_CSV = BASE_DIR / "class_balance.csv"

MODEL_FILE = BASE_DIR / "best_model.joblib"
DATA_FILE = BASE_DIR / "data_clean.csv"

LABEL_TO_TEXT = {2: "Home Win", 1: "Draw", 0: "Away Win"}
TEXT_TO_LABEL = {"Home Win": 2, "Draw": 1, "Away Win": 0}

@st.cache_data
def load_csv(path) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_history(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df

def team_outcome_from_ftr(row, team):
    ftr = row["FTR"]
    if ftr == 1:
        return "D"
    if ftr == 2:
        return "W" if row["HomeTeam"] == team else "L"
    if ftr == 0:
        return "W" if row["AwayTeam"] == team else "L"
    return None

def weighted_wdl_last5(data, team, asof_date, recency_weights=(1, 2, 3, 4, 5)):
    past = data[
        ((data["HomeTeam"] == team) | (data["AwayTeam"] == team)) &
        (data["Date"] < asof_date)
    ].tail(5)

    if len(past) < 5:
        return None

    rw = np.array(recency_weights, dtype=float)
    rw = rw / rw.sum()

    wW = wD = wL = 0.0
    for i, (_, row) in enumerate(past.iterrows()):
        outcome = team_outcome_from_ftr(row, team)
        if outcome == "W":
            wW += rw[i]
        elif outcome == "D":
            wD += rw[i]
        elif outcome == "L":
            wL += rw[i]

    weighted_points = 3 * wW + 1 * wD
    return (wW, wD, wL, weighted_points)

def rolling_avg_last5(data, team, asof_date, home_col, away_col):
    past = data[
        ((data["HomeTeam"] == team) | (data["AwayTeam"] == team)) &
        (data["Date"] < asof_date)
    ].tail(5)

    if len(past) < 5:
        return None

    vals = []
    for _, r in past.iterrows():
        if r["HomeTeam"] == team:
            vals.append(r[home_col])
        else:
            vals.append(r[away_col])

    vals = [v for v in vals if pd.notna(v)]
    if len(vals) == 0:
        return None
    return float(np.mean(vals))

def build_features_for_fixture(history_df, home_team, away_team, asof_date):
    hw = weighted_wdl_last5(history_df, home_team, asof_date)
    aw = weighted_wdl_last5(history_df, away_team, asof_date)
    if hw is None or aw is None:
        return None

    feat = {
        "HomeW5": hw[0], "HomeD5": hw[1], "HomeL5": hw[2], "HomeWeightedPoints5": hw[3],
        "AwayW5": aw[0], "AwayD5": aw[1], "AwayL5": aw[2], "AwayWeightedPoints5": aw[3],
    }

    pairs = [
        ("HomeAvgHST5", home_team, "HST", "AST"),
        ("AwayAvgAST5", away_team, "HST", "AST"),
        ("HomeAvgHC5", home_team, "HC", "AC"),
        ("AwayAvgAC5", away_team, "HC", "AC"),
        ("HomeAvgHF5", home_team, "HF", "AF"),
        ("AwayAvgAF5", away_team, "HF", "AF"),
        ("HomeAvgHY5", home_team, "HY", "AY"),
        ("AwayAvgAY5", away_team, "HY", "AY"),
        ("HomeAvgHR5", home_team, "HR", "AR"),
        ("AwayAvgAR5", away_team, "HR", "AR"),
    ]

    for name, team, home_col, away_col in pairs:
        val = rolling_avg_last5(history_df, team, asof_date, home_col, away_col)
        if val is None:
            return None
        feat[name] = val

    return pd.DataFrame([feat])

page = st.sidebar.radio("Navigate", ["Overview", "Model Performance", "Predict a Match"])

st.title("Football Match Outcome Prediction Dashboard")

if page == "Overview":
    st.subheader("What this dashboard shows")
    st.markdown(
        """
- **Class balance** (Home / Draw / Away)  
- **Cross-validation vs test** performance across multiple ML models  
- **Interactive prediction** for a selected Home vs Away team using your trained model  
        """
    )

    if BAL_CSV.exists():
        bal = load_csv(BAL_CSV)
        st.subheader("Class Balance")
        if "Count" in bal.columns and ("FTR" in bal.columns or "FTR_Label" in bal.columns):
            name_col = "FTR" if "FTR" in bal.columns else "FTR_Label"
            fig = px.pie(bal, names=name_col, values="Count", title="Class Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(bal, use_container_width=True)
    else:
        st.info("class_balance.csv not found yet. Export it from your notebook to show class balance here.")

elif page == "Model Performance":
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cross-Validation (Macro F1)")
        if CV_CSV.exists():
            cv_df = load_csv(CV_CSV).sort_values("CV_MacroF1", ascending=False)
            fig = px.bar(cv_df, x="Model", y="CV_MacroF1", title="Cross-Validation Macro F1 by Model")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(cv_df, use_container_width=True)
        else:
            st.info("cv_results.csv not found yet. Save your CV summary to CSV to display it here.")

    with col2:
        st.subheader("Hold-out Test (Macro F1)")
        if TEST_CSV.exists():
            test_df = load_csv(TEST_CSV).sort_values("Test_MacroF1", ascending=False)
            fig = px.bar(test_df, x="Model", y="Test_MacroF1", title="Test Macro F1 by Model")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(test_df, use_container_width=True)
        else:
            st.info("test_results.csv not found yet. Save your test summary to CSV to display it here.")

    st.divider()
    st.subheader("Add your other figures")

    for img in ["figure_class_balance.png", "figure_forecasting_timeline.png", "figure_pipeline_clean_correct.png"]:
        img_path = BASE_DIR / img
        if img_path.exists():
            st.image(str(img_path), caption=img, use_container_width=True)

elif page == "Predict a Match":
    st.subheader("Predict an Upcoming Fixture")

    if not MODEL_FILE.exists():
        st.error(f"Model file not found: {MODEL_FILE.name}. Upload it next to your Streamlit app.")
        st.stop()

    if not DATA_FILE.exists():
        st.error(f"Historical data file not found: {DATA_FILE.name}. Upload it next to your Streamlit app.")
        st.stop()

    model = load_model(MODEL_FILE)
    history = load_history(DATA_FILE)

    if history["FTR"].dtype == object:
        map_ftr = {"H": 2, "D": 1, "A": 0}
        history["FTR"] = history["FTR"].map(map_ftr)

    teams = sorted(pd.unique(pd.concat([history["HomeTeam"], history["AwayTeam"]], ignore_index=True)).tolist())

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        home_team = st.selectbox("Home team", teams, index=0)
    with c2:
        away_team = st.selectbox("Away team", teams, index=1 if len(teams) > 1 else 0)
    with c3:
        asof_date = st.date_input("Prediction date (before kickoff)", value=history["Date"].max().date())

    if home_team == away_team:
        st.warning("Home and Away teams must be different.")
        st.stop()

    asof_date = pd.to_datetime(asof_date)

    st.caption("Features are computed from the previous 5 matches for each team (strictly before the selected date).")

    if st.button("Predict outcome"):
        X_pred = build_features_for_fixture(history, home_team, away_team, asof_date)

        if X_pred is None:
            st.error("Not enough historical matches (need at least 5 prior matches for both teams with required stats). Try an earlier date or different teams.")
            st.stop()

        feature_cols = [
            "HomeW5", "HomeD5", "HomeL5", "HomeWeightedPoints5",
            "AwayW5", "AwayD5", "AwayL5", "AwayWeightedPoints5",
            "HomeAvgHST5", "AwayAvgAST5",
            "HomeAvgHC5", "AwayAvgAC5",
            "HomeAvgHF5", "AwayAvgAF5",
            "HomeAvgHY5", "AwayAvgAY5",
            "HomeAvgHR5", "AwayAvgAR5"
        ]
        X_pred = X_pred[feature_cols]

        pred = model.predict(X_pred)[0]
        pred_text = LABEL_TO_TEXT.get(int(pred), str(pred))
        st.success(f"Predicted outcome: **{pred_text}**")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_pred)[0]
            classes = getattr(model, "classes_", [0, 1, 2])
            prob_df = pd.DataFrame({"Class": classes, "Probability": probs})
            prob_df["Label"] = prob_df["Class"].map(LABEL_TO_TEXT)
            prob_df = prob_df.sort_values("Probability", ascending=False)

            fig = px.bar(prob_df, x="Label", y="Probability", title="Prediction Probabilities")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(prob_df[["Label", "Probability"]], use_container_width=True)

        with st.expander("Show engineered input features"):
            st.dataframe(X_pred, use_container_width=True)
