import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Compliance Monitoring Dashboard")
st.subheader("Fairness audit: graduation-rate predictions across Pell groups")

## loading files
merged = pd.read_csv('merged_ipeds.csv')
model = joblib.load('logreg_model.pkl')

## monitoring
features = ['accept_rate', 'UPGRNTP']
X = merged[features]
y = merged['high_grad']
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)

st.metric("Overall accuracy", f"{accuracy_score(y_test, y_pred):.1%}")

from fairlearn.metrics import MetricFrame
from sklearn.metrics import recall_score

st.subheader("Metrics by Pell group")

sensitive = merged.loc[X_test.index, 'pell_group']
mf = MetricFrame(
    metrics={'accuracy': accuracy_score, 'recall': recall_score},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive
)
st.dataframe(mf.by_group.style.format("{:.1%}"))
