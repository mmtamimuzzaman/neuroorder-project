"""
==========================================================
features.py
Author : Md. Jakaria Masud

Project : NeuroOrder
Task    : Feature Engineering

This file extracts features from every candidate move.

Feature Vector:
[
    piece_difference,
    center_control,
    player_threats,
    opponent_threats,
    mobility
]
==========================================================
"""

from game_engine import (
    make_move,
    get_valid_moves
)

# -------------------------------------------------------
# Count total pieces of a player
# -------------------------------------------------------
def count_pieces(board, player):
    count = 0

    for row in board:
        for cell in row:
            if cell == player:
                count += 1

    return count


# -------------------------------------------------------
# Feature 1
# Piece Difference
# -------------------------------------------------------
def piece_difference(board, player):

    opponent = 2 if player == 1 else 1

    my_piece = count_pieces(board, player)
    opp_piece = count_pieces(board, opponent)

    return my_piece - opp_piece


# -------------------------------------------------------
# Feature 2
# Center Control
# -------------------------------------------------------
def center_control(board, player):

    cols = len(board[0])

    center = cols // 2

    score = 0

    for row in board:

        if row[center] == player:
            score += 1

    return score


# -------------------------------------------------------
# Utility Function
# Count player & empty cells
# -------------------------------------------------------
def analyze_window(window, player):

    opponent = 2 if player == 1 else 1

    player_count = window.count(player)
    opponent_count = window.count(opponent)
    empty_count = window.count(0)

    return player_count, opponent_count, empty_count


# -------------------------------------------------------
# Feature 3
# Player Threats
#
# Threat =
# 3 own pieces
# 1 empty cell
# -------------------------------------------------------
def player_threats(board, player):

    rows = len(board)
    cols = len(board[0])

    threat = 0

    # Horizontal
    for r in range(rows):

        for c in range(cols-3):

            window = [
                board[r][c+i]
                for i in range(4)
            ]

            p,o,e = analyze_window(window,player)

            if p==3 and e==1:
                threat +=1

    # Vertical
    for c in range(cols):

        for r in range(rows-3):

            window = [
                board[r+i][c]
                for i in range(4)
            ]

            p,o,e = analyze_window(window,player)

            if p==3 and e==1:
                threat +=1

    # Positive Diagonal
    for r in range(rows-3):

        for c in range(cols-3):

            window = [
                board[r+i][c+i]
                for i in range(4)
            ]

            p,o,e = analyze_window(window,player)

            if p==3 and e==1:
                threat +=1

    # Negative Diagonal
    for r in range(3,rows):

        for c in range(cols-3):

            window = [
                board[r-i][c+i]
                for i in range(4)
            ]

            p,o,e = analyze_window(window,player)

            if p==3 and e==1:
                threat +=1

    return threat


# -------------------------------------------------------
# Feature 4
# Opponent Threats
# -------------------------------------------------------
def opponent_threats(board, player):

    opponent = 2 if player==1 else 1

    return player_threats(board, opponent)


# -------------------------------------------------------
# Feature 5
# Mobility
#
# Number of legal moves
# -------------------------------------------------------
def mobility(board):

    return len(get_valid_moves(board))


# -------------------------------------------------------
# Main Feature Function
#
# This is the only function Abrar will call.
# -------------------------------------------------------
def extract_features(board, move, player):

    # Apply candidate move
    new_board = make_move(board, move, player)

    feature_vector = [

        piece_difference(
            new_board,
            player
        ),

        center_control(
            new_board,
            player
        ),

        player_threats(
            new_board,
            player
        ),

        opponent_threats(
            new_board,
            player
        ),

        mobility(
            new_board
        )

    ]

    return feature_vector


# -------------------------------------------------------
# Testing
# -------------------------------------------------------
if __name__ == "__main__":

    from game_engine import create_board

    board = create_board()

    move = 3

    player = 1

    features = extract_features(
        board,
        move,
        player
    )

    print(features)