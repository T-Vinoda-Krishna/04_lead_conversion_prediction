# Lead Conversion Prediction

## Business problem
Sales teams have limited capacity. Which leads should be prioritized for follow-up to maximize conversions?

## ML objective
Predict probability of lead conversion.

## Workflow
Data generation → EDA → preprocessing → train/test split → Logistic Regression + Random Forest → evaluation → threshold analysis → explainability.

## Demo
```bash
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```
