import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "piece_difference",
    "center_control",
    "player_threats",
    "opponent_threats",
    "mobility",
]

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")
test_df = pd.read_csv("test.csv")

X_train, y_train = train_df[FEATURES], train_df["label"]
X_val, y_val = val_df[FEATURES], val_df["label"]
X_test, y_test = test_df[FEATURES], test_df["label"]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=30,
        random_state=42
    ))
])

model.fit(X_train, y_train)

for name, X, y in [
    ("Validation", X_val, y_val),
    ("Test", X_test, y_test),
]:
    pred = model.predict(X)
    print(f"{name} R²: {r2_score(y, pred):.6f}")
    print(f"{name} MAE: {mean_absolute_error(y, pred):.6f}")

joblib.dump(model, "neuroorder_model.pkl")
print("Saved: neuroorder_model.pkl")
