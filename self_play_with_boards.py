import csv
import math
import random

from game_engine import (
    create_board,
    get_valid_moves,
    make_move,
    check_winner,
    minimax_ab,
    minimax_ab_value,
    WIN_SCORE,
)
from features import extract_features

TOTAL_GAMES = 5000
SEARCH_DEPTH = 4
OUTPUT_FILE = "self_play_data_with_boards.csv"
EVAL_SCALE = 20.0

random.seed(42)

def serialize_board(board):
    return "/".join("".join(map(str, row)) for row in board)

def generate_label(board, move, player):
    next_board = make_move(board, move, player)
    opponent = 2 if player == 1 else 1

    _, _, score = minimax_ab_value(
        next_board,
        SEARCH_DEPTH - 1,
        -math.inf,
        math.inf,
        opponent
    )

    if player == 2:
        score = -score

    if score >= WIN_SCORE:
        return 1.0
    if score <= -WIN_SCORE:
        return -1.0

    score = score / EVAL_SCALE
    score = max(-1.0, min(1.0, score))
    return round(score, 6)

def save_candidates(board, player, writer, game_id):
    valid_moves = get_valid_moves(board)

    for move in valid_moves:
        features = extract_features(board, move, player)
        label = generate_label(board, move, player)

        writer.writerow([
            game_id,
            serialize_board(board),
            player,
            move,
            features[0],
            features[1],
            features[2],
            features[3],
            features[4],
            label,
        ])

def play_game(writer, game_id):
    board = create_board()
    player = 1
    winner = None

    while winner is None:
        valid_moves = get_valid_moves(board)
        if not valid_moves:
            break

        save_candidates(board, player, writer, game_id)

        # Preserve Jakaria's current 20% random / 80% Alpha-Beta policy.
        if random.random() < 0.20:
            move = random.choice(valid_moves)
        else:
            move, _ = minimax_ab(
                board,
                SEARCH_DEPTH,
                -math.inf,
                math.inf,
                player
            )

        board = make_move(board, move, player)
        winner = check_winner(board)
        player = 2 if player == 1 else 1

def generate_dataset():
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "game_id", "board", "player", "move",
            "piece_difference", "center_control",
            "player_threats", "opponent_threats",
            "mobility", "label"
        ])

        for game_id in range(TOTAL_GAMES):
            play_game(writer, game_id)

            if (game_id + 1) % 100 == 0:
                print(f"{game_id + 1}/{TOTAL_GAMES} Games Completed")

    print(f"\nDataset generated: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_dataset()
