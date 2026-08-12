# NeuroOrder — corrected end-to-end pipeline

Run in this order:

1. `python self_play_with_boards.py`
2. `python split_dataset.py`
3. `python train_model.py`
4. `python evaluate.py`

Important:
- Dataset labels use the backed-up minimax value, not `1/nodes`.
- Each row stores board/player/move so held-out positions can be reconstructed.
- Train/validation/test are split by `game_id` to prevent game-level leakage.
- The model trains only on the five requested features.
- NeuroOrder predicts all legal moves in one batch.
- `evaluate.py` compares No Ordering, Killer-Move, History Heuristic, and NeuroOrder.
- The final benchmark must be run after regenerating the dataset; old feature-only CSVs cannot supply the original boards.
