import math
import random

ROWS = 6
COLS = 6
WIN_LENGTH = 4
WIN_SCORE = 100_000


def create_board(rows=ROWS, cols=COLS):
    return [[0] * cols for _ in range(rows)]


def get_valid_moves(board):
    cols = len(board[0])
    return [c for c in range(cols) if board[0][c] == 0]


def make_move(board, col, player):
    new_board = [row[:] for row in board]
    rows = len(new_board)
    for r in range(rows - 1, -1, -1):
        if new_board[r][col] == 0:
            new_board[r][col] = player
            return new_board
    raise ValueError(f"Column {col} is already full")


def _count_direction(board, r, c, dr, dc, player):
    rows, cols = len(board), len(board[0])
    count = 0
    r, c = r + dr, c + dc
    while 0 <= r < rows and 0 <= c < cols and board[r][c] == player:
        count += 1
        r, c = r + dr, c + dc
    return count


def check_winner(board):
    rows, cols = len(board), len(board[0])
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for r in range(rows):
        for c in range(cols):
            player = board[r][c]
            if player == 0:
                continue
            for dr, dc in directions:
                if _count_direction(board, r, c, dr, dc, player) >= WIN_LENGTH - 1:
                    return player
    if not get_valid_moves(board):
        return 0
    return None


def print_board(board):
    symbols = {0: ".", 1: "X", 2: "O"}
    for row in board:
        print(" ".join(symbols[v] for v in row))
    print(" ".join(str(c) for c in range(len(board[0]))))
    print()


def evaluate(board):
    cols = len(board[0])
    center_col = cols // 2
    score = 0
    for row in board:
        if row[center_col] == 1:
            score += 3
        elif row[center_col] == 2:
            score -= 3
    return score


def minimax_ab(board, depth, alpha, beta, player, ordering_fn=None, on_cutoff=None):
    best_move, nodes, _score = _minimax(board, depth, alpha, beta, player, ordering_fn, on_cutoff)
    return best_move, nodes


def minimax_ab_value(board, depth, alpha, beta, player, ordering_fn=None, on_cutoff=None):
    """Extra helper (not in the original shared contract): same search, but ALSO
    returns the backed-up minimax value/score (always from Player 1's perspective,
    positive = good for P1, negative = good for P2). Needed for value-based labels."""
    return _minimax(board, depth, alpha, beta, player, ordering_fn, on_cutoff)


def _minimax(board, depth, alpha, beta, player, ordering_fn, on_cutoff):
    nodes = 1
    winner = check_winner(board)
    if winner is not None:
        if winner == 0:
            return None, nodes, 0
        score = (WIN_SCORE + depth) if winner == 1 else -(WIN_SCORE + depth)
        return None, nodes, score

    if depth == 0:
        return None, nodes, evaluate(board)

    moves = get_valid_moves(board)
    if not moves:
        return None, nodes, 0

    if ordering_fn:
        moves = ordering_fn(board, moves, player)

    best_move = moves[0]
    opponent = 2 if player == 1 else 1
    maximizing = (player == 1)
    best_score = -math.inf if maximizing else math.inf

    for move in moves:
        child = make_move(board, move, player)
        _, child_nodes, score = _minimax(child, depth - 1, alpha, beta, opponent, ordering_fn, on_cutoff)
        nodes += child_nodes
        if maximizing:
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, best_score)
        else:
            if score < best_score:
                best_score, best_move = score, move
            beta = min(beta, best_score)
        if beta <= alpha:
            if on_cutoff:
                on_cutoff(depth, move, player)
            break

    return best_move, nodes, best_score


class KillerMoveHeuristic:
    def __init__(self, max_killers_per_depth=2):
        self.killers = {}
        self.max_killers_per_depth = max_killers_per_depth

    def order(self, board, moves, player):
        all_killers = []
        for depth_killers in self.killers.values():
            for m in depth_killers:
                if m not in all_killers:
                    all_killers.append(m)
        ordered = [m for m in all_killers if m in moves]
        ordered += [m for m in moves if m not in ordered]
        return ordered

    def record(self, depth, move, player):
        lst = self.killers.setdefault(depth, [])
        if move in lst:
            lst.remove(move)
        lst.insert(0, move)
        if len(lst) > self.max_killers_per_depth:
            lst.pop()


class HistoryHeuristic:
    def __init__(self):
        self.table = {}

    def order(self, board, moves, player):
        return sorted(moves, key=lambda m: self.table.get(m, 0), reverse=True)

    def record(self, depth, move, player):
        self.table[move] = self.table.get(move, 0) + depth * depth


_killer_tracker = KillerMoveHeuristic()
_history_tracker = HistoryHeuristic()


def killer_move_ordering(board, moves, player):
    return _killer_tracker.order(board, moves, player)


def history_heuristic_ordering(board, moves, player):
    return _history_tracker.order(board, moves, player)


def killer_move_search(board, depth, alpha, beta, player):
    return minimax_ab(board, depth, alpha, beta, player,
                       ordering_fn=killer_move_ordering, on_cutoff=_killer_tracker.record)


def history_heuristic_search(board, depth, alpha, beta, player):
    return minimax_ab(board, depth, alpha, beta, player,
                       ordering_fn=history_heuristic_ordering, on_cutoff=_history_tracker.record)
