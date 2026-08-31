import streamlit as st, pandas as pd, joblib
from pathlib import Path

ROOT = Path(__file__).parent
df = pd.read_csv(ROOT/"data/leads.csv")
model_path = ROOT/"model.joblib"

st.title("Lead Conversion Predictor")
st.caption("Synthetic CRM dataset. Run src/train.py once before opening the app.")

if not model_path.exists():
    st.warning("Model not found. Run: python src/train.py")
    st.stop()

model = joblib.load(model_path)
row = {}
for c in df.drop(columns=["lead_id","converted"]).columns:
    if df[c].dtype == "object":
        row[c] = st.selectbox(c, sorted(df[c].unique()))
    else:
        row[c] = st.number_input(c, float(df[c].median()))

if st.button("Predict conversion probability"):
    x = pd.DataFrame([row])
    p = float(model.predict_proba(x)[0,1])
    st.metric("Predicted conversion probability", f"{p:.1%}")
    st.info("Use the score as a prioritization aid, not as an automatic decision.")
