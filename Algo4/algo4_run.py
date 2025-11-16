import sys
import os
import time
import csv
import tracemalloc
import random
import copy
from tabulate import tabulate
from typing import List, Dict, Tuple

MAX_STEPS = 200000
N = 9

# Metrics Tracker (same fields as Algo 1–3)
class MetricsTracker:
    def __init__(self):
        self.nodes = 0
        self.backtracks = 0

# Min-Conflicts Helper Functions
def get_conflicts(board, row, col, val):
    conflicts = 0
    # Row
    for k in range(N):
        if k != col and board[row][k] == val:
            conflicts += 1
    # Column
    for k in range(N):
        if k != row and board[k][col] == val:
            conflicts += 1
    # Subgrid
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(start_row, start_row + 3):
        for c in range(start_col, start_col + 3):
            if (r != row or c != col) and board[r][c] == val:
                conflicts += 1
    return conflicts


def random_initial_board(puzzle):
    board = copy.deepcopy(puzzle)
    for i in range(N):
        present = [x for x in board[i] if x != 0]
        missing = [n for n in range(1, 10) if n not in present]
        random.shuffle(missing)
        midx = 0
        for j in range(N):
            if board[i][j] == 0:
                board[i][j] = missing[midx]
                midx += 1
    return board


def conflicted_cells(board, fixed):
    cells = []
    for i in range(N):
        for j in range(N):
            if not fixed[i][j]:
                if get_conflicts(board, i, j, board[i][j]) > 0:
                    cells.append((i, j))
    return cells


def total_conflicts_for_swap(board, r1, c1, r2, c2):
    v1, v2 = board[r1][c1], board[r2][c2]
    return get_conflicts(board, r1, c1, v2) + get_conflicts(board, r2, c2, v1)


# Min-Conflicts Solver (nodes + backtracks integrated)
def min_conflicts_solver(puzzle, tracker, max_steps=MAX_STEPS):
    fixed = [[puzzle[i][j] != 0 for j in range(N)] for i in range(N)]
    board = random_initial_board(puzzle)

    for step in range(max_steps):
        tracker.nodes += 1  # Each loop = 1 node

        conflicted = conflicted_cells(board, fixed)
        if not conflicted:
            return board, step

        row, col = random.choice(conflicted)

        best_swap, best_conf = None, None
        for j in range(N):
            if j == col or fixed[row][j]:
                continue
            conf = total_conflicts_for_swap(board, row, col, row, j)
            if best_conf is None or conf < best_conf:
                best_conf, best_swap = conf, j

        if best_swap is not None:
            board[row][col], board[row][best_swap] = (
                board[row][best_swap],
                board[row][col],
            )
        else:
            # Fallback → treat as backtrack
            tracker.backtracks += 1

            min_conf = float("inf")
            best_vals = []
            for val in range(1, 10):
                c = get_conflicts(board, row, col, val)
                if c < min_conf:
                    min_conf = c
                    best_vals = [val]
                elif c == min_conf:
                    best_vals.append(val)
            board[row][col] = random.choice(best_vals)

    return board, max_steps

# Utilities
def parse_puzzle_string(p):
    return [list(map(int, p[i:i+9])) for i in range(0, 81, 9)]


def board_to_string(board):
    return "".join(str(x) for row in board for x in row)


def print_puzzle(puzzle_string: str):
    print()
    for r in range(9):
        row = [
            puzzle_string[r * 9 + c] if puzzle_string[r * 9 + c] != '0' else '.'
            for c in range(9)
        ]
        print(" ".join(row[:3]) + " | " +
              " ".join(row[3:6]) + " | " +
              " ".join(row[6:9]))
        if r in [2, 5]:
            print("-" * 21)
    print()


def is_valid_solution(board):
    # Rows
    for r in range(9):
        if set(board[r]) != set(range(1, 10)):
            return False
    # Columns
    for c in range(9):
        if set(board[r][c] for r in range(9)) != set(range(1, 10)):
            return False
    # 3x3 subgrids
    for br in range(3):
        for bc in range(3):
            vals = [
                board[r][c]
                for r in range(br * 3, br * 3 + 3)
                for c in range(bc * 3, bc * 3 + 3)
            ]
            if set(vals) != set(range(1, 10)):
                return False
    return True

# Solve a single puzzle
def solve_single_puzzle(puzzle_string: str, show_output=True):
    if show_output:
        print("ORIGINAL PUZZLE:")
        print_puzzle(puzzle_string)

    tracker = MetricsTracker()
    puzzle = parse_puzzle_string(puzzle_string)

    tracemalloc.start()
    start = time.perf_counter()

    success = False
    solution = None
    steps = 0

    try:
        solution, steps = min_conflicts_solver(puzzle, tracker, MAX_STEPS)
        success = is_valid_solution(solution)
    except Exception:
        success = False

    end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    runtime = end - start
    memory_mb = peak / (1024 * 1024)

    if show_output and success:
        print("\nSOLVED PUZZLE:")
        print_puzzle(board_to_string(solution))

    return (
        board_to_string(solution) if success else None,
        runtime,
        memory_mb,
        tracker.nodes,
        tracker.backtracks,
        success,
        False, 
    )

# File reader + CSV
def read_puzzles_from_file(filename):
    puzzles = []
    with open(filename, "r", encoding="utf-8") as f:
        digits = "".join(ch for ch in f.read() if ch.isdigit())
    for i in range(0, len(digits), 81):
        if i + 81 <= len(digits):
            puzzles.append(digits[i:i+81])
    return puzzles


def log_to_csv(row, csv_filename="performance_log_min_conflict.csv"):
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, "a", newline="", encoding="utf-8") as f:
        fieldnames = [
            "File", "PuzzleIndex", "GlobalIndex",
            "Runtime(s)", "Memory(MB)",
            "NodesVisited", "Backtracks",
            "Success", "TimedOut"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# Solve ALL puzzles 
def solve_all_puzzles(filenames: List[str]):

    print(f"\n{'='*70}")
    print("SUDOKU SOLVER – Min-Conflicts Local Search")
    print("Metrics included: NodesVisited, Backtracks")
    print(f"{'='*70}\n")

    log_file = "performance_log_min_conflict.csv"
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"Cleared existing {log_file}\n")

    all_results = []
    global_index = 0

    for filename in filenames:
        puzzles = read_puzzles_from_file(filename)
        if not puzzles:
            continue

        print(f"\nReading file: {filename}")
        print(f"Found {len(puzzles)} puzzle(s)\n")

        for idx, puzzle in enumerate(puzzles, start=1):
            global_index += 1
            print(f"\n{'='*70}")
            print(f"File: {filename} | Puzzle #{idx} | Global #{global_index}")
            print(f"{'='*70}")

            solved, runtime, mem, nodes, backs, success, timed_out = \
                solve_single_puzzle(puzzle, show_output=True)

            status = "✓" if success else ("⏱️" if timed_out else "✗")

            row = {
                "File": os.path.basename(filename),
                "PuzzleIndex": idx,
                "GlobalIndex": global_index,
                "Runtime(s)": f"{runtime:.6f}",
                "Memory(MB)": f"{mem:.6f}",
                "NodesVisited": nodes,
                "Backtracks": backs,
                "Success": success,
                "TimedOut": timed_out 
            }

            log_to_csv(row, log_file)
            all_results.append(row)

    # SUMMARY REPORT 
    print(f"\n{'='*70}")
    print("SUMMARY REPORT")
    print(f"{'='*70}\n")

    table_data = []
    for r in all_results:
        status = "✓" if r["Success"] else ("⏱️" if r["TimedOut"] else "✗")
        table_data.append([
            r["GlobalIndex"], r["File"], r["PuzzleIndex"],
            r["Runtime(s)"], r["Memory(MB)"],
            r["NodesVisited"], r["Backtracks"], status
        ])

    print(tabulate(
        table_data,
        headers=["Global#", "File", "Puzzle#", "Runtime(s)",
                 "Memory(MB)", "Nodes", "Backtracks", "Status"],
        tablefmt="grid"
    ))

    # STATISTICS BY DIFFICULTY 
    print(f"\n{'='*70}")
    print("STATISTICS BY DIFFICULTY (with Accuracy)")
    print(f"{'='*70}\n")

    files = {}
    for r in all_results:
        fname = r["File"]
        if fname not in files:
            files[fname] = {
                "solved": 0, "total": 0,
                "runtime": 0.0, "nodes": 0, "backs": 0
            }

        files[fname]["total"] += 1
        if r["Success"]:
            files[fname]["solved"] += 1
            files[fname]["runtime"] += float(r["Runtime(s)"])
            files[fname]["nodes"] += int(r["NodesVisited"])
            files[fname]["backs"] += int(r["Backtracks"])

    stats_table = []
    for fname, s in files.items():
        total = s["total"]
        solved = s["solved"]

        acc = (solved / total * 100) if total else 0
        avg_runtime = s["runtime"] / solved if solved else 0
        avg_nodes = s["nodes"] / solved if solved else 0
        avg_backs = s["backs"] / solved if solved else 0

        stats_table.append([
            fname,
            f"{solved}/{total}",
            f"{acc:.1f}%",
            f"{avg_runtime:.6f}",
            f"{avg_nodes:.0f}",
            f"{avg_backs:.0f}"
        ])

    print(tabulate(
        stats_table,
        headers=["File", "Solved", "Accuracy",
                 "Avg Runtime(s)", "Avg Nodes", "Avg Backtracks"],
        tablefmt="grid"
    ))

    # OVERALL STATISTICS
    print(f"\n{'='*70}")
    print("OVERALL STATISTICS")
    print(f"{'='*70}\n")

    total = len(all_results)
    solved = sum(1 for r in all_results if r["Success"])
    acc = (solved / total * 100) if total else 0

    overall = [
        ["Total Puzzles", total],
        ["Solved", solved],
        ["Failed", total - solved],
        ["Overall Accuracy", f"{acc:.2f}%"]
    ]
    print(tabulate(overall, tablefmt="simple"))

    print(f"\n{'='*70}")
    print(f"Results saved to: performance_log_min_conflict.csv")
    print(f"Algorithm: Minimum Conflict Local Search")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 algo4_run.py easy.txt medium.txt hard.txt")
        sys.exit(1)
    solve_all_puzzles(sys.argv[1:])
