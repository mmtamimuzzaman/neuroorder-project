# NeuroOrder

### Learning Move-Ordering Policies to Improve the Search Efficiency of Alpha-Beta Pruning in Adversarial Game Search

**CSE366 — Artificial Intelligence, Section 2 · East West University**
Submitted to **Dr. Tania Sultana**, Assistant Professor, Dept. of CSE

[![Course](https://img.shields.io/badge/course-CSE366-blue)]()
[![Status](https://img.shields.io/badge/status-complete-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-yellow)]()
[![License](https://img.shields.io/badge/license-academic-lightgrey)]()

---

## What is NeuroOrder?

Alpha-Beta Pruning is the classical optimization of Minimax search, but its efficiency depends almost entirely on **the order in which moves are explored**. Under near-optimal ordering, the effective branching factor drops from `b` to roughly `√b`; under poor ordering, pruning barely helps at all.

For decades, the standard fix has been hand-crafted heuristics — **killer-move** and **history heuristic** — fixed rules that don't learn or adapt. **NeuroOrder replaces that hand-crafted rule with a small, trained neural network** that learns to rank candidate moves from the search's own self-play data, and plugs into Alpha-Beta purely as the move-ordering step. Because Alpha-Beta is guaranteed to return the same final decision regardless of move order, NeuroOrder can be benchmarked safely and fairly — purely on **search efficiency** (nodes expanded, wall-clock time, effective branching factor) — against a no-ordering baseline and the two classical heuristics, on a reduced 6×6 Connect-4 test bed.

A live, interactive, in-browser demo of the whole system is included — see [`neuroorder_live_demo.html`](#-live-demo).

---

## The Research Gap

Before designing NeuroOrder, we surveyed the literature at the intersection of neural networks and tree search and found a consistent pattern:

| Line of work | What it learns | What it misses |
|---|---|---|
| Neural A\* / learned heuristics (Yonetani et al. 2021; Kim & An 2020) | A cost estimate for single-agent pathfinding | Restricted to grid worlds / single-agent search; both papers explicitly flag generalization as future work |
| Neural Minimax/Alpha-Beta enhancements (Ma 2023; AlphaZero-style systems) | The **board-evaluation function** (how good a position is) | Almost never targets **move-ordering** — which move to look at first |
| MCTS/RL comparisons on Connect-4 | Overall move/board evaluation | Doesn't isolate move-ordering as a standalone learning target |

**The gap:** move-ordering is widely acknowledged as the single largest lever on Alpha-Beta's practical efficiency, yet a dedicated, lightweight, *learned* move-ordering policy — evaluated in isolation from the evaluation function, with correctness held fixed — is largely unaddressed in the literature we reviewed.

**Our research question:**
> Can a small neural network, trained on move-quality labels derived from self-play, learn a move-ordering policy that reduces the number of nodes Alpha-Beta must expand — relative to killer-move and history heuristics — without ever changing the algorithm's final decision?

Full literature review and citations are in the [project proposal](#-references).

---

## System Architecture

```
┌──────────────────────┐     ┌────────────────────┐     ┌──────────────────────┐
│  1. Self-Play Engine   │ --> │ 2. Feature Extractor │ --> │ 3. NeuroOrder Model    │
│  (game_engine.py)      │     │  (features.py)        │     │  (train_model.py)      │
│  Baseline Alpha-Beta   │     │  5 hand-crafted        │     │  scikit-learn MLP      │
│  generates true        │     │  board/move features   │     │  (32,16) hidden units   │
│  minimax-backed labels │     │                        │     │                        │
└──────────────────────┘     └────────────────────┘     └──────────┬───────────┘
                                                                       │
                                                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  4. Alpha-Beta Search Engine — same engine, pluggable ordering_fn          │
│     No Ordering │ Killer-Move │ History Heuristic │ NeuroOrder (this step) │
└───────────────────────────────────┬────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  5. Evaluation Harness (evaluate.py) — nodes expanded, time, branching     │
│     factor, correctness check, feature importance, robustness ablations   │
└──────────────────────────────────────────────────────────────────────────┘
```

NeuroOrder never touches the evaluation function or final decision — it only changes **which sibling move Alpha-Beta looks at first**.

---

## Repository Structure

```
neuroorder-project/
├── game_engine.py                  # Board, Minimax, Alpha-Beta, killer-move & history heuristics
├── self_play_with_boards.py        # Self-play data generator (records board + player + move)
├── self_play_data_with_boards.csv  # Generated self-play dataset
├── split_dataset.py                # Game-level train/validation/test split (no leakage)
├── train.csv / validation.csv / test.csv
├── features.py                     # extract_features(board, move, player) -> 5-dim vector
├── train_model.py                  # Trains the NeuroOrder MLP on the labelled dataset
├── neuroorder_model.pkl            # Trained model (StandardScaler + MLPRegressor pipeline)
├── neuroorder_ordering.py          # Wraps the trained model as an ordering_fn
├── evaluate.py                     # Runs and compares all 4 ordering conditions
├── neuroorder_live_demo.html       # Interactive live demo — see below
└── README.md
```

---

## Shared Contract

Every module is built against the same interfaces, so the four ordering strategies are interchangeable:

**Board representation** — 6×6 2D list, `0` = empty, `1` = Player 1 (Max), `2` = Player 2 (Min).

```python
# game_engine.py
get_valid_moves(board) -> list[int]
make_move(board, col, player) -> new_board
check_winner(board) -> 0 / 1 / 2 / None
minimax_ab(board, depth, alpha, beta, player, ordering_fn=None) -> (best_move, nodes_expanded)
```

`ordering_fn` is any function `(board, moves, player) -> sorted_moves`. Killer-move, history heuristic, and NeuroOrder are each just a different `ordering_fn` plugged into the exact same Alpha-Beta engine — the search code itself is written once.

```python
# features.py
extract_features(board, move, player) -> [piece_diff, center_control, threats, opponent_threats, mobility]
```

---

## Corrected Pipeline

> As of the latest revision, the pipeline generates labels from the true **backed-up minimax value** (not a proxy like `1/nodes`), and every training row stores the originating `(board, player, move, game_id)` so any held-out position can be reconstructed exactly for evaluation. Splits are made **at the game level**, never the position level, to prevent leakage between near-duplicate positions from the same game.

Run the full pipeline in order:

```bash
# 1. Generate self-play data (records full board state per position, not just features)
python self_play_with_boards.py

# 2. Split into train / validation / test — split by game_id, not by row
python split_dataset.py

# 3. Train the NeuroOrder neural network
python train_model.py

# 4. Run the full 4-way comparison and produce metrics + graphs
python evaluate.py
```

**Notes on the corrected pipeline:**
- Labels are the actual minimax-backed value from the baseline search, giving the network a real quality signal rather than a weak proxy.
- Each dataset row keeps enough information (`board`, `player`, `move`, `game_id`) to reconstruct the exact position later — required because `evaluate.py` needs the original boards, not just precomputed feature vectors.
- The model is trained strictly on the five agreed-upon features — no leakage of raw board state into the network.
- At inference time, NeuroOrder scores **all legal moves for a position in a single batched prediction** rather than one-by-one, keeping ordering overhead low relative to the pruning it buys back.
- `evaluate.py` benchmarks **No Ordering, Killer-Move, History Heuristic, and NeuroOrder** side by side on the same held-out test positions.
- Any time the dataset is regenerated, the benchmark must be re-run from scratch — an old feature-only CSV cannot supply the original boards the corrected pipeline needs.

To sanity-check the engine alone:
```bash
python game_engine.py     # runs the built-in self-tests (win detection, draws, correctness across depths)
```

---

## Live Demo

`neuroorder_live_demo.html` is a **fully self-contained, single-file** interactive demo — no server, no install, works offline once opened. It reimplements the entire engine (board, Minimax, Alpha-Beta, both classical heuristics, and the *actual trained* NeuroOrder network weights) directly in the browser, verified node-for-node against this repository's Python implementation.

- Play against the AI, or watch full AI-vs-AI self-play
- Live, animated node-count "race" between all four ordering conditions on every move
- A live view into NeuroOrder's predicted score for each legal column
- A running correctness check confirming all four methods agree on the optimal move
- A cumulative leaderboard tracking total nodes and speedup across the whole game

Open it directly in any modern browser (Chrome/Edge/Firefox) — ideal for presentations and demos.

---

## Evaluation Metrics

| Metric | What it shows |
|---|---|
| **Nodes expanded** | Primary efficiency metric — how much of the tree each method had to visit |
| **Wall-clock search time** | Practical speed under identical hardware/depth |
| **Effective branching factor** | `nodes^(1/depth)` — how close each method gets to the theoretical `√b` best case |
| **Correctness check** | Confirms all four conditions return the *same final move* — ordering must never change the outcome |
| **Feature importance** (permutation importance) | Which of the five features NeuroOrder actually relies on |
| **Robustness** | Same experiment repeated across ≥2 search depths and ≥2 board sizes |

## Results

*(Fill in after running `evaluate.py` on your machine — numbers depend on hardware and the trained model snapshot.)*

| Condition | Avg. Nodes Expanded | Avg. Time (ms) | Speedup vs. No Ordering | Final Move Match |
|---|---|---|---|---|
| No Ordering | | | 1.0× (baseline) | ✓ |
| Killer-Move | | | | ✓ |
| History Heuristic | | | | ✓ |
| **NeuroOrder** | | | | ✓ |

Graphs (`evaluate.py` output) go in a `results/` folder — bar charts for nodes expanded, search time, and feature importance per condition.

---

## Tools & Technologies

- **Language:** Python 3.10+
- **ML Library:** scikit-learn (`MLPRegressor`, `StandardScaler`, `Pipeline`, `permutation_importance`)
- **Supporting Libraries:** NumPy, Pandas, Matplotlib
- **Live Demo:** Vanilla HTML/CSS/JavaScript (no build step, no dependencies)
- **Version Control:** Git / GitHub
- **Hardware:** Runs on a standard laptop — no GPU required by design

## Setup

```bash
git clone https://github.com/mmtamimuzzaman/neuroorder-project.git
cd neuroorder-project

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install numpy pandas scikit-learn matplotlib
```

---

## Team

| Member | Student ID | Responsibility |
|---|---|---|
| **M. M Tamim Uz Zaman** | 2023-3-60-298 | Baseline game engine · Minimax & Alpha-Beta Pruning · killer-move & history heuristics |
| **Md. Jakaria Masud** | 2023-3-60-386 | Self-play data generation · feature engineering · game-level dataset splitting |
| **Abrar Tahsan** | 2023-3-60-118 | NeuroOrder neural network design, training, hyperparameter tuning, and integration |
| **Md. Razwan Ahamed** | 2023-3-60-418 | Experimental evaluation · feature-importance analysis · visualization · report |

All members jointly participated in the literature review, experimental design, risk-mitigation planning, and the final presentation.

---

## Expected Outcomes

Because Alpha-Beta's correctness never depends on move order, this comparison is safe to run and easy to verify regardless of which strategy wins. Either outcome is a legitimate, quantified finding:

- **If NeuroOrder matches or beats the classical heuristics** — it demonstrates that a lightweight, learned policy can rival decades-old hand-crafted rules on search efficiency, and the feature-importance analysis shows whether it converges on the same intuitions classical AI theory already values (e.g., center control, threat detection).
- **If it doesn't** — that's still a meaningful, well-instrumented negative result about the limits of a small MLP trained on a modest self-play dataset, which the risk-analysis section of the proposal accounts for from the outset.

## References

1. Yonetani, R., Taniai, T., Barekatain, M., Nishimura, M., Kanezaki, A. (2021). *Path Planning using Neural A\* Search.* ICML 2021, PMLR 139:12029–12039.
2. Kim, S., An, B. (2020). *Learning Heuristic A\*: Efficient Graph Search using Neural Network.* ICRA 2020, pp. 9542–9547.
3. *Learning Empirically Admissible Neural Heuristics for Combinatorial Search.* arXiv:2606.04860.
4. Ma, W. (2023). *Optimization of Alpha-Beta Pruning Based on Heuristic Algorithm.*
5. *Analysis of Game Tree Search Algorithms Using Minimax Algorithm and Alpha-Beta Pruning.* ResearchGate.
6. *An Evolutionary Framework for Connect-4 as Test-Bed for Comparison of Advanced Minimax, Q-Learning and MCTS.* arXiv:2405.16595.
7. Russell, S., Norvig, P. (2021). *Artificial Intelligence: A Modern Approach*, 4th Ed. Pearson.
8. Knuth, D. E., Moore, R. W. (1975). *An Analysis of Alpha-Beta Pruning.* Artificial Intelligence, 6(4), 293–326.

---

## License

Academic project submitted for **CSE366 (Artificial Intelligence), Section 2**, East West University. Shared for coursework and educational reference.

---

<p align="center">Built with Alpha-Beta Pruning, a small neural network, and way too much coffee — Tamim · Jakaria · Abrar · Razwan</p>
