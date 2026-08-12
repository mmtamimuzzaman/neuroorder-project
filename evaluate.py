import ast
import math
import time
import pandas as pd

from game_engine import (
    minimax_ab,
    killer_move_search,
    history_heuristic_search,
)
from neuroorder_ordering import neuroorder_ordering

DEPTH = 5
TEST_FILE = "test.csv"

def deserialize_board(s):
    # Format: "000000/000000/..."
    return [list(map(int, row)) for row in s.split("/")]

def run_one(board, player, method):
    start = time.perf_counter()

    if method == "No Ordering":
        move, nodes = minimax_ab(
            board, DEPTH, -math.inf, math.inf, player
        )
    elif method == "Killer-Move":
        move, nodes = killer_move_search(
            board, DEPTH, -math.inf, math.inf, player
        )
    elif method == "History Heuristic":
        move, nodes = history_heuristic_search(
            board, DEPTH, -math.inf, math.inf, player
        )
    elif method == "NeuroOrder":
        move, nodes = minimax_ab(
            board, DEPTH, -math.inf, math.inf, player,
            ordering_fn=neuroorder_ordering
        )
    else:
        raise ValueError(method)

    return move, nodes, time.perf_counter() - start

df = pd.read_csv(TEST_FILE)

# One row exists for each candidate move. Deduplicate to obtain actual
# held-out board/player positions.
positions = df[["game_id", "board", "player"]].drop_duplicates().reset_index(drop=True)

methods = [
    "No Ordering",
    "Killer-Move",
    "History Heuristic",
    "NeuroOrder",
]

results = {m: {"nodes": [], "time": [], "moves": []} for m in methods}

for i, row in positions.iterrows():
    board = deserialize_board(row["board"])
    player = int(row["player"])

    print(f"Position {i+1}/{len(positions)}")

    for method in methods:
        move, nodes, elapsed = run_one(board, player, method)
        results[method]["nodes"].append(nodes)
        results[method]["time"].append(elapsed)
        results[method]["moves"].append(move)

print("\n===== FINAL TEST RESULTS =====")
for method in methods:
    avg_nodes = sum(results[method]["nodes"]) / len(results[method]["nodes"])
    avg_time = sum(results[method]["time"]) / len(results[method]["time"])
    print(f"{method:20s} Average Nodes: {avg_nodes:10.2f}  Average Time: {avg_time:.6f}s")

# Correctness check: all methods should return the same minimax move
baseline = results["No Ordering"]["moves"]
for method in methods[1:]:
    same = sum(a == b for a, b in zip(baseline, results[method]["moves"]))
    total = len(baseline)
    print(f"{method:20s} Same best move: {same}/{total} ({100*same/total:.2f}%)")
