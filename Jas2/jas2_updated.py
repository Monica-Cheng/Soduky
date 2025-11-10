#!/usr/bin/env python3
"""
sudoku_runner_fc_minimal.py

Wrapper for Backtracking + Forward Checking + MRV implementation
Source: https://github.com/paccionesawyer/sudokuSolver-CSP

Algorithm: Backtracking + Forward Checking + MRV Heuristic

Cross-platform version (Windows + macOS + Linux)
✔ Replaced signal.SIGALRM with threading.Timer
✔ Same CSV + statistics logic
✔ No changes to solving algorithm
"""

import sys
import os
import time
import csv
import tracemalloc
import threading
from tabulate import tabulate
from typing import List, Dict, Tuple
from io import StringIO

# Import original modules
from SudokuBoard import SudokuBoard
from solver import forward_checking, minimum_remaining_values

TIMEOUT_SECONDS = 30
timeout_occurred = False


# ===============================================================
# Timeout utilities
# ===============================================================
def set_timeout_flag():
    """Triggered when timer expires."""
    global timeout_occurred
    timeout_occurred = True


# ===============================================================
# Single puzzle solver
# ===============================================================
def solve_single_puzzle(puzzle_string: str, show_output: bool = True) -> Tuple:
    """
    Solve using ORIGINAL Forward Checking + MRV implementation.
    Returns: (solution, runtime, memory_mb, nodes, backtracks, success, timed_out)
    """
    global timeout_occurred
    timeout_occurred = False

    if len(puzzle_string) != 81:
        raise ValueError(f"Puzzle must be 81 characters, got {len(puzzle_string)}")

    # Print puzzle
    if show_output:
        print("ORIGINAL PUZZLE:")
        print_puzzle(puzzle_string)

    success = False
    solution = None

    # Start timeout timer
    timer = threading.Timer(TIMEOUT_SECONDS, set_timeout_flag)
    timer.start()

    # Start tracking
    tracemalloc.start()
    start_time = time.perf_counter()

    try:
        # Construct board from string
        file_obj = StringIO(puzzle_string)
        puzzle = SudokuBoard(file_obj)

        if show_output:
            print("\nSolving with Backtracking + Forward Checking + MRV...")

        # Solve using original function
        success = forward_checking(puzzle, minimum_remaining_values)

        # Stop timer
        timer.cancel()

        # If solved before timeout
        if not timeout_occurred and success:
            solution = board_to_string(puzzle)
            if show_output:
                print("\nSOLVED PUZZLE:")
                puzzle.print_board()

    except Exception as e:
        success = False
        if show_output:
            print(f"\n⚠️ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    finally:
        timer.cancel()

    # Stop tracking
    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    runtime = end_time - start_time
    peak_mb = peak / (1024 * 1024)

    # Get built-in metrics
    nodes = getattr(puzzle, "unique_states", 0)
    backtracks = getattr(puzzle, "backtracks", 0)

    # Timeout detection
    if timeout_occurred:
        success = False
        if show_output:
            print(f"\n⏱️ TIMEOUT: Exceeded {TIMEOUT_SECONDS} seconds")

    return solution, runtime, peak_mb, nodes, backtracks, success, timeout_occurred


# ===============================================================
# Utility functions
# ===============================================================
def board_to_string(puzzle) -> str:
    """Convert SudokuBoard back to 81-character string"""
    return "".join(str(puzzle.board[i][j]) for i in range(9) for j in range(9))


def print_puzzle(puzzle_string: str) -> None:
    """Pretty-print puzzle"""
    print()
    for i in range(9):
        row = []
        for j in range(9):
            idx = i * 9 + j
            val = puzzle_string[idx] if puzzle_string[idx] != '0' else '.'
            row.append(val)
        print(" ".join(row[0:3]) + " | " +
              " ".join(row[3:6]) + " | " +
              " ".join(row[6:9]))
        if i in [2, 5]:
            print("-" * 21)
    print()


def read_puzzles_from_file(filename: str) -> List[str]:
    """Read puzzles from text file"""
    puzzles = []
    if not os.path.isfile(filename):
        print(f"⚠️ File not found: {filename}")
        return puzzles

    with open(filename, "r", encoding="utf-8") as f:
        digits_only = "".join(ch for ch in f.read() if ch.isdigit())
        if len(digits_only) % 81 != 0:
            print(f"⚠️ {filename} has {len(digits_only)} digits, not multiple of 81")
        for i in range(0, len(digits_only), 81):
            if i + 81 <= len(digits_only):
                puzzles.append(digits_only[i:i+81])
    return puzzles


def log_to_csv(row: Dict, csv_filename: str = "performance_log_fc_minimal.csv") -> None:
    """Append results to CSV"""
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["File", "PuzzleIndex", "GlobalIndex", "Runtime(s)",
                      "Memory(MB)", "NodesVisited", "Backtracks",
                      "Success", "TimedOut"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


# ===============================================================
# Batch solver
# ===============================================================
def solve_all_puzzles(filenames: List[str]) -> None:
    """Run solver on all files"""
    print(f"\n{'='*70}")
    print("SUDOKU SOLVER - Backtracking + Forward Checking + MRV")
    print("Source: github.com/paccionesawyer/sudokuSolver-CSP")
    print(f"Timeout: {TIMEOUT_SECONDS}s per puzzle")
    print(f"{'='*70}\n")

    log_file = "performance_log_fc_minimal.csv"
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"Cleared existing {log_file}\n")

    all_results = []
    global_index = 0

    for filename in filenames:
        print(f"\n{'='*70}")
        print(f"Reading file: {filename}")
        print(f"{'='*70}")
        puzzles = read_puzzles_from_file(filename)
        if not puzzles:
            continue
        print(f"Found {len(puzzles)} puzzle(s) in {filename}\n")

        for puzzle_idx, puzzle in enumerate(puzzles, start=1):
            global_index += 1
            print(f"\n{'='*70}")
            print(f"File: {filename} | Puzzle #{puzzle_idx} | Global #{global_index}")
            print(f"{'='*70}")

            try:
                solved, runtime, memory_mb, nodes, backtracks, success, timed_out = \
                    solve_single_puzzle(puzzle, show_output=True)

                row = {
                    "File": os.path.basename(filename),
                    "PuzzleIndex": puzzle_idx,
                    "GlobalIndex": global_index,
                    "Runtime(s)": f"{runtime:.6f}",
                    "Memory(MB)": f"{memory_mb:.6f}",
                    "NodesVisited": nodes,
                    "Backtracks": backtracks,
                    "Success": success,
                    "TimedOut": timed_out
                }
                log_to_csv(row, log_file)
                all_results.append(row)

                if success:
                    print(f"✓ Solved in {runtime:.6f}s | Nodes: {nodes}, Backtracks: {backtracks}")
                elif timed_out:
                    print(f"⏱️ Timed out after {runtime:.6f}s | Nodes: {nodes}")
                else:
                    print(f"✗ Failed after {runtime:.6f}s | Nodes: {nodes}, Backtracks: {backtracks}")

            except Exception as e:
                print(f"⚠️ ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
                row = {
                    "File": os.path.basename(filename),
                    "PuzzleIndex": puzzle_idx,
                    "GlobalIndex": global_index,
                    "Runtime(s)": "0.000000",
                    "Memory(MB)": "0.000000",
                    "NodesVisited": 0,
                    "Backtracks": 0,
                    "Success": False,
                    "TimedOut": False
                }
                log_to_csv(row, log_file)
                all_results.append(row)

    # ===========================================================
    # Summary report
    # ===========================================================
    if not all_results:
        print("No results to display.")
        return

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
    print(tabulate(table_data,
                   headers=["Global#", "File", "Puzzle#", "Runtime(s)",
                            "Memory(MB)", "Nodes", "Backtracks", "Status"],
                   tablefmt="grid"))

    print(f"\n{'='*70}")
    print("STATISTICS BY DIFFICULTY (with Accuracy)")
    print(f"{'='*70}\n")

    files = {}
    for r in all_results:
        fname = r["File"]
        if fname not in files:
            files[fname] = {"solved": 0, "total": 0, "timed_out": 0,
                            "total_runtime": 0.0, "total_nodes": 0, "total_backtracks": 0}
        files[fname]["total"] += 1
        if r["Success"]:
            files[fname]["solved"] += 1
            files[fname]["total_runtime"] += float(r["Runtime(s)"])
            files[fname]["total_nodes"] += int(r["NodesVisited"])
            files[fname]["total_backtracks"] += int(r["Backtracks"])
        if r["TimedOut"]:
            files[fname]["timed_out"] += 1

    stats_table = []
    for fname, s in files.items():
        total, solved, timed = s["total"], s["solved"], s["timed_out"]
        acc = (solved / total * 100) if total > 0 else 0
        avg_runtime = s["total_runtime"] / solved if solved > 0 else 0
        avg_nodes = s["total_nodes"] / solved if solved > 0 else 0
        avg_back = s["total_backtracks"] / solved if solved > 0 else 0
        stats_table.append([fname, f"{solved}/{total}", f"{acc:.1f}%", timed,
                            f"{avg_runtime:.6f}", f"{avg_nodes:.0f}", f"{avg_back:.0f}"])

    print(tabulate(stats_table,
                   headers=["File", "Solved", "Accuracy", "TimedOut",
                            "Avg Runtime(s)", "Avg Nodes", "Avg Backtracks"],
                   tablefmt="grid"))

    print(f"\n{'='*70}")
    print("OVERALL STATISTICS")
    print(f"{'='*70}\n")

    total_puzzles = len(all_results)
    total_solved = sum(1 for r in all_results if r["Success"])
    total_timed = sum(1 for r in all_results if r["TimedOut"])
    acc = (total_solved / total_puzzles * 100) if total_puzzles > 0 else 0
    overall = [
        ["Total Puzzles", total_puzzles],
        ["Solved", total_solved],
        ["Failed", total_puzzles - total_solved - total_timed],
        ["Timed Out", total_timed],
        ["Overall Accuracy", f"{acc:.2f}%"]
    ]
    print(tabulate(overall, tablefmt="simple"))

    print(f"\n{'='*70}")
    print(f"Results saved to: {log_file}")
    print(f"Algorithm: Backtracking + Forward Checking + MRV")
    print(f"Timeout: {TIMEOUT_SECONDS}s per puzzle")
    print(f"{'='*70}\n")


# ===============================================================
# Entry point
# ===============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 sudoku_runner_fc_minimal.py <file1.txt> [file2.txt] ...")
        sys.exit(1)

    input_files = sys.argv[1:]
    solve_all_puzzles(input_files)
