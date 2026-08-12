import pandas as pd
from pathlib import Path

INPUT = "self_play_data_with_boards.csv"

TRAIN_OUT = "train.csv"
VALID_OUT = "validation.csv"
TEST_OUT = "test.csv"

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
RANDOM_STATE = 42

df = pd.read_csv(INPUT)

# Split by game_id, not by row, so candidate moves from the same game
# cannot leak across train/validation/test.
game_ids = df["game_id"].drop_duplicates().sample(frac=1, random_state=RANDOM_STATE).tolist()

n = len(game_ids)
train_end = int(n * TRAIN_RATIO)
valid_end = train_end + int(n * VALID_RATIO)

train_ids = set(game_ids[:train_end])
valid_ids = set(game_ids[train_end:valid_end])
test_ids = set(game_ids[valid_end:])

train = df[df.game_id.isin(train_ids)].copy()
valid = df[df.game_id.isin(valid_ids)].copy()
test = df[df.game_id.isin(test_ids)].copy()

# Keep the board/player/move columns in the evaluation files, while training
# will explicitly use only the five feature columns.
train.to_csv(TRAIN_OUT, index=False)
valid.to_csv(VALID_OUT, index=False)
test.to_csv(TEST_OUT, index=False)

print("Dataset split completed.")
print(f"Train:      {len(train)} rows, {len(train_ids)} games")
print(f"Validation: {len(valid)} rows, {len(valid_ids)} games")
print(f"Test:       {len(test)} rows, {len(test_ids)} games")
