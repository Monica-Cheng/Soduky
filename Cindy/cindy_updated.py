#!/usr/bin/env python3
"""
sudoku_runner_min_conflict.py

Minimum Conflict Local Search Algorithm for Sudoku
- Works on all platforms (no SIGALRM, uses threading.Timer)
- Tracks runtime, memory, iterations (nodes)
- Outputs CSV and statistics summary
"""

import sys
import os
import time
import csv
import tracemalloc
import random
import copy
import threading
from tabulate import tabulate
from typing import List, Dict, Tuple

TIMEOUT_SECONDS = 30
MAX_STEPS = 200000
N = 9
timeout_occurred = False


# ============================================================
# Timeout handling
# ============================================================
def set_timeout_flag():
    global timeout_occurred
    timeout_occurred = True


# ============================================================
# Min-Conflicts Algorithm
# ============================================================
def get_conflicts(board, row, col, val):
    conflicts = 0
    for k in range(N):
        if k != col and board[row][k] == val:
            conflicts += 1
    for k in range(N):
        if k != row and board[k][col] == val:
            conflicts += 1
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
    conflicts = []
    for i in range(N):
        for j in range(N):
            if not fixed[i][j]:
                if get_conflicts(board, i, j, board[i][j]) > 0:
                    conflicts.append((i, j))
    return conflicts


def total_conflicts_for_swap(board, r1, c1, r2, c2):
    v1, v2 = board[r1][c1], board[r2][c2]
    return get_conflicts(board, r1, c1, v2) + get_conflicts(board, r2, c2, v1)


def min_conflicts_solver(puzzle, max_steps=MAX_STEPS):
    fixed = [[puzzle[i][j] != 0 for j in range(N)] for i in range(N)]
    board = random_initial_board(puzzle)

    for step in range(max_steps):
        if timeout_occurred:
            break

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
            board[row][col], board[row][best_swap] = board[row][best_swap], board[row][col]
        else:
            min_conf = float("inf")
            best_vals = []
            for val in range(1, 10):
                c = get_conflicts(board, row, col, val)
                if c < min_conf:
                    min_conf, best_vals = c, [val]
                elif c == min_conf:
                    best_vals.append(val)
            board[row][col] = random.choice(best_vals)

    return board, max_steps


def is_valid_solution(board):
    for r in range(9):
        if set(board[r]) != set(range(1, 10)):
            return False
    for c in range(9):
        if set(board[r][c] for r in range(9)) != set(range(1, 10)):
            return False
    for br in range(3):
        for bc in range(3):
            vals = [board[r][c] for r in range(br*3, br*3+3)
                    for c in range(bc*3, bc*3+3)]
            if set(vals) != set(range(1, 10)):
                return False
    return True


# ============================================================
# Helper functions
# ============================================================
def parse_puzzle_string(puzzle_string: str):
    if len(puzzle_string) != 81:
        raise ValueError("Puzzle must be 81 characters.")
    nums = [int(ch) for ch in puzzle_string]
    return [nums[i:i+9] for i in range(0, 81, 9)]


def board_to_string(board):
    return ''.join(str(val) for row in board for val in row)


def print_puzzle(puzzle_string: str):
    print()
    for i in range(9):
        row = []
        for j in range(9):
            idx = i * 9 + j
            val = puzzle_string[idx] if puzzle_string[idx] != '0' else '.'
            row.append(val)
        print(" ".join(row[0:3]) + " | " + " ".join(row[3:6]) + " | " + " ".join(row[6:9]))
        if i in [2, 5]:
            print("-" * 21)
    print()


# ============================================================
# Main puzzle runner (with timeout)
# ============================================================
def solve_single_puzzle(puzzle_string: str, show_output=True):
    global timeout_occurred
    timeout_occurred = False

    if show_output:
        print("ORIGINAL PUZZLE:")
        print_puzzle(puzzle_string)

    puzzle = parse_puzzle_string(puzzle_string)
    success = False
    solution_board = None
    steps = 0

    timer = threading.Timer(TIMEOUT_SECONDS, set_timeout_flag)
    timer.start()

    tracemalloc.start()
    start = time.perf_counter()

    try:
        solution_board, steps = min_conflicts_solver(puzzle, MAX_STEPS)
        success = is_valid_solution(solution_board)
    except Exception as e:
        print(f"⚠️ ERROR: {e}")
        success = False
    finally:
        timer.cancel()

    end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    runtime = end - start
    memory_mb = peak / (1024 * 1024)

    if timeout_occurred:
        success = False
        if show_output:
            print(f"\n⏱️ TIMEOUT after {TIMEOUT_SECONDS}s")

    if show_output and success:
        print("SOLVED PUZZLE:")
        print_puzzle(board_to_string(solution_board))
    elif show_output and not success and not timeout_occurred:
        print("\n⚠️  Max iterations reached without valid solution")

    return (
        board_to_string(solution_board) if success else None,
        runtime, memory_mb, steps, 0, success, timeout_occurred
    )


# ============================================================
# File reading / logging / batch processing
# ============================================================
def read_puzzles_from_file(filename):
    puzzles = []
    if not os.path.isfile(filename):
        print(f"⚠️ File not found: {filename}")
        return puzzles
    with open(filename, "r", encoding="utf-8") as f:
        digits = "".join(ch for ch in f.read() if ch.isdigit())
        for i in range(0, len(digits), 81):
            if i + 81 <= len(digits):
                puzzles.append(digits[i:i+81])
    return puzzles


def log_to_csv(row: Dict, csv_filename="performance_log_min_conflict.csv"):
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["File", "PuzzleIndex", "GlobalIndex", "Runtime(s)",
                      "Memory(MB)", "NodesVisited", "Backtracks", "Success", "TimedOut"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def solve_all_puzzles(filenames: List[str]):
    print(f"\n{'='*70}")
    print("SUDOKU SOLVER - Minimum Conflict Local Search")
    print("Algorithm: Min-Conflicts with Row-Swap Repair")
    print(f"Timeout: {TIMEOUT_SECONDS}s per puzzle")
    print(f"Max Steps: {MAX_STEPS}")
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
        print(f"\nFile: {filename} | {len(puzzles)} puzzles found")

        for idx, puzzle in enumerate(puzzles, 1):
            global_index += 1
            print(f"\n{'='*70}")
            print(f"File: {filename} | Puzzle #{idx} | Global #{global_index}")
            print(f"{'='*70}")

            solved, runtime, mem, steps, backs, success, timeout = solve_single_puzzle(puzzle, show_output=True)

            row = {
                "File": os.path.basename(filename),
                "PuzzleIndex": idx,
                "GlobalIndex": global_index,
                "Runtime(s)": f"{runtime:.6f}",
                "Memory(MB)": f"{mem:.6f}",
                "NodesVisited": steps,
                "Backtracks": backs,
                "Success": success,
                "TimedOut": timeout
            }
            log_to_csv(row, log_file)
            all_results.append(row)

    # Summary report
    print(f"\n{'='*70}")
    print("SUMMARY REPORT")
    print(f"{'='*70}\n")

    if not all_results:
        print("No puzzles processed.")
        return

    table = []
    for r in all_results:
        status = "✓" if r["Success"] else ("⏱️" if r["TimedOut"] else "✗")
        table.append([
            r["GlobalIndex"], r["File"], r["PuzzleIndex"], r["Runtime(s)"],
            r["Memory(MB)"], r["NodesVisited"], r["Backtracks"], status
        ])
    print(tabulate(table,
                   headers=["Global#", "File", "Puzzle#", "Runtime(s)", "Memory(MB)",
                            "Steps", "Backtracks", "Status"],
                   tablefmt="grid"))

    # Stats by difficulty
    print(f"\n{'='*70}")
    print("STATISTICS BY DIFFICULTY (with Accuracy)")
    print(f"{'='*70}\n")

    files = {}
    for r in all_results:
        f = r["File"]
        if f not in files:
            files[f] = {"solved": 0, "total": 0, "timed": 0,
                        "runtime": 0.0, "nodes": 0}
        files[f]["total"] += 1
        if r["Success"]:
            files[f]["solved"] += 1
            files[f]["runtime"] += float(r["Runtime(s)"])
            files[f]["nodes"] += int(r["NodesVisited"])
        if r["TimedOut"]:
            files[f]["timed"] += 1

    stats = []
    for f, s in files.items():
        acc = (s["solved"] / s["total"] * 100) if s["total"] > 0 else 0
        avg_runtime = s["runtime"] / s["solved"] if s["solved"] > 0 else 0
        avg_nodes = s["nodes"] / s["solved"] if s["solved"] > 0 else 0
        stats.append([f, f"{s['solved']}/{s['total']}", f"{acc:.1f}%", s["timed"],
                      f"{avg_runtime:.6f}", f"{avg_nodes:.0f}"])

    print(tabulate(stats,
                   headers=["File", "Solved", "Accuracy", "TimedOut",
                            "Avg Runtime(s)", "Avg Steps"],
                   tablefmt="grid"))

    # Overall stats
    print(f"\n{'='*70}")
    print("OVERALL STATISTICS")
    print(f"{'='*70}\n")

    total = len(all_results)
    solved = sum(1 for r in all_results if r["Success"])
    timed = sum(1 for r in all_results if r["TimedOut"])
    acc = (solved / total * 100) if total else 0
    overall = [
        ["Total Puzzles", total],
        ["Solved", solved],
        ["Failed", total - solved - timed],
        ["Timed Out", timed],
        ["Overall Accuracy", f"{acc:.2f}%"]
    ]
    print(tabulate(overall, tablefmt="simple"))
    print(f"\n{'='*70}")
    print(f"Results saved to: {log_file}")
    print(f"Algorithm: Minimum Conflict Local Search")
    print(f"Note: 'NodesVisited' = steps, 'Backtracks' = 0")
    print(f"{'='*70}\n")


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sudoku_runner_min_conflict.py easy.txt medium.txt hard.txt")
        sys.exit(1)

    solve_all_puzzles(sys.argv[1:])
