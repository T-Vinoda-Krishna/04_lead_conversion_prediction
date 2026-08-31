from pathlib import Path
import joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/leads.csv")
X = df.drop(columns=["lead_id","converted"])
y = df["converted"]

cat = X.select_dtypes(include=["object"]).columns.tolist()
num = [c for c in X.columns if c not in cat]

pre = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),("scale", StandardScaler())]), num),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat)
])

models = {
    "logistic": Pipeline([("pre",pre),("model",LogisticRegression(max_iter=2000))]),
    "random_forest": Pipeline([("pre",pre),("model",RandomForestClassifier(n_estimators=350, random_state=42, class_weight="balanced"))])
}

Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
best = None
for name,m in models.items():
    m.fit(Xtr,ytr)
    p = m.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte,p)
    print(name, "ROC-AUC:", round(auc,4))
    if best is None or auc > best[0]:
        best = (auc,name,m)

print("Best:", best[1])
joblib.dump(best[2], ROOT / "model.joblib")
