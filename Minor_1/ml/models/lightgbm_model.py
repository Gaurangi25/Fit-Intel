from ml.models.train_model import (
    X_train, X_test,
    y_train_mood, y_test_mood,
    y_train_prod, y_test_prod,
    feature_names,
    w_train
)

from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge

from lightgbm import LGBMRegressor   # ✅ CHANGED

import json
from datetime import datetime
import joblib
import os
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =======================
# 🔥 Model Config (LightGBM)
# =======================

def get_model():
    return LGBMRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

# =======================
# 🔹 Mood Model
# =======================

mood_model = get_model()
mood_model.fit(X_train, y_train_mood, sample_weight=w_train)

pred_mood = mood_model.predict(X_test)

r2_mood = r2_score(y_test_mood, pred_mood)
rmse_mood = np.sqrt(mean_squared_error(y_test_mood, pred_mood))
mae_mood = mean_absolute_error(y_test_mood, pred_mood)

print("\nMood Model Results")
print("R2:", r2_mood)
print("RMSE:", rmse_mood)
print("MAE:", mae_mood)


# =======================
# 🔹 Productivity Model
# =======================

prod_model = get_model()
prod_model.fit(X_train, y_train_prod, sample_weight=w_train)

pred_prod = prod_model.predict(X_test)

r2_prod = r2_score(y_test_prod, pred_prod)
rmse_prod = np.sqrt(mean_squared_error(y_test_prod, pred_prod))
mae_prod = mean_absolute_error(y_test_prod, pred_prod)

print("\nProductivity Model Results")
print("R2:", r2_prod)
print("RMSE:", rmse_prod)
print("MAE:", mae_prod)


# =======================
# 🔹 Cross Validation
# =======================

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

mood_scores = cross_val_score(
    get_model(),
    X_train,
    y_train_mood,
    cv=cv,
    scoring="r2"
)

prod_scores = cross_val_score(
    get_model(),
    X_train,
    y_train_prod,
    cv=cv,
    scoring="r2"
)

print("\nMood CV Average:", mood_scores.mean())
print("Productivity CV Average:", prod_scores.mean())


# =======================
# 🔹 Baseline (Ridge)
# =======================

ridge = Ridge()

ridge.fit(X_train, y_train_mood)
print("\nBaseline Mood R2:", r2_score(y_test_mood, ridge.predict(X_test)))

ridge.fit(X_train, y_train_prod)
print("Baseline Productivity R2:", r2_score(y_test_prod, ridge.predict(X_test)))


# =======================
# 🔹 Feature Importance
# =======================

print("\nFeature Importance")

importance = mood_model.feature_importances_

for name, score in sorted(
    zip(feature_names, importance),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{name}: {score:.4f}")


# =======================
# 🔹 Save Models
# =======================

model_dir = os.path.join(BASE_DIR, "saved_models")
os.makedirs(model_dir, exist_ok=True)

joblib.dump(mood_model, os.path.join(model_dir, "lightgbm_mood.pkl"))
joblib.dump(prod_model, os.path.join(model_dir, "lightgbm_productivity.pkl"))

print("\nModels saved successfully")


# =======================
# 🔹 Save Results
# =======================

results_dir = os.path.join(BASE_DIR, "results")
os.makedirs(results_dir, exist_ok=True)

json_file = os.path.join(results_dir, "model_results.json")

results_data = {
    "model": "LightGBM",
    "timestamp": datetime.now().isoformat(),
    "mood": {
        "test_r2": float(r2_mood),
        "cv_r2_mean": float(mood_scores.mean())
    },
    "productivity": {
        "test_r2": float(r2_prod),
        "cv_r2_mean": float(prod_scores.mean())
    }
}

if os.path.exists(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
else:
    data = {}

data["LightGBM"] = results_data

with open(json_file, "w") as f:
    json.dump(data, f, indent=4)

print("Results saved successfully")