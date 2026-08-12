import joblib
import pandas as pd
from features import extract_features

FEATURES = [
    "piece_difference",
    "center_control",
    "player_threats",
    "opponent_threats",
    "mobility",
]

model = joblib.load("neuroorder_model.pkl")

def neuroorder_ordering(board, moves, player):
    if not moves:
        return []

    rows = [extract_features(board, move, player) for move in moves]
    X = pd.DataFrame(rows, columns=FEATURES)

    # Batch prediction: one model call for all candidate moves.
    scores = model.predict(X)

    # Higher predicted value = better move for the current player.
    return [
        move for move, _ in
        sorted(zip(moves, scores), key=lambda x: x[1], reverse=True)
    ]
