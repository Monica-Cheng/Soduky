#!/usr/bin/env python3
"""
sudoku_runner_min_conflict.py

Minimum Conflict Local Search Algorithm for Sudoku

Algorithm: Min-Conflicts with Row-Swap Repair
- Creates complete initial assignment (each row is permutation of 1-9)
- Iteratively swaps values within rows to minimize conflicts
- NO backtracking - pure local search

MINIMAL CHANGES:
1. Read from easy.txt, medium.txt, hard.txt
2. Track metrics (steps = nodes, no backtracks)
3. Output to CSV in your format
4. Add timeout support

ORIGINAL LOGIC UNCHANGED
"""

import sys
import os
import time
import csv
import tracemalloc
import signal
import random
import copy
from tabulate import tabulate
from typing import List, Dict, Tuple

TIMEOUT_SECONDS = 30
MAX_STEPS = 200000  # Maximum iterations per puzzle
N = 9


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Timeout")


# ============================================================
# ORIGINAL Min-Conflicts Algorithm (unchanged)
# ============================================================

def get_conflicts(board, row, col, val):
    """Count conflicts that would occur if cell (row,col) had value val."""
    conflicts = 0
    # row conflicts
    for k in range(N):
        if k != col and board[row][k] == val:
            conflicts += 1
    # column conflicts
    for k in range(N):
        if k != row and board[k][col] == val:
            conflicts += 1
    # subgrid conflicts
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(start_row, start_row + 3):
        for c in range(start_col, start_col + 3):
            if (r != row or c != col) and board[r][c] == val:
                conflicts += 1
    return conflicts


def random_initial_board(puzzle):
    """Fill each row's zeros with missing numbers (row permutation)."""
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
    """Return list of non-fixed cells with conflicts."""
    conflicts = []
    for i in range(N):
        for j in range(N):
            if not fixed[i][j]:
                if get_conflicts(board, i, j, board[i][j]) > 0:
                    conflicts.append((i, j))
    return conflicts


def total_conflicts_for_swap(board, r1, c1, r2, c2):
    """Compute conflicts if two cells were swapped."""
    v1, v2 = board[r1][c1], board[r2][c2]
    c_after = get_conflicts(board, r1, c1, v2) + get_conflicts(board, r2, c2, v1)
    return c_after


def min_conflicts_solver(puzzle, max_steps=MAX_STEPS):
    """
    ORIGINAL Min-Conflicts solver with row-swap repair.
    Returns: (solution_board, steps_taken)
    """
    fixed = [[puzzle[i][j] != 0 for j in range(N)] for i in range(N)]
    board = random_initial_board(puzzle)

    for step in range(max_steps):
        conflicted = conflicted_cells(board, fixed)
        if not conflicted:
            return board, step
        
        # Pick random conflicted cell
        row, col = random.choice(conflicted)

        # Find best swap within same row
        best_swap = None
        best_conf = None
        for j in range(N):
            if j == col or fixed[row][j]:
                continue
            conf = total_conflicts_for_swap(board, row, col, row, j)
            if best_conf is None or conf < best_conf:
                best_conf = conf
                best_swap = j

        # Perform swap or fallback
        if best_swap is not None:
            board[row][col], board[row][best_swap] = board[row][best_swap], board[row][col]
        else:
            # Fallback: try all values
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


def is_valid_solution(board):
    """Check if solution is valid (no conflicts)."""
    # Check rows
    for r in range(9):
        if set(board[r]) != set(range(1, 10)):
            return False
    # Check columns
    for c in range(9):
        if set(board[r][c] for r in range(9)) != set(range(1, 10)):
            return False
    # Check boxes
    for br in range(3):
        for bc in range(3):
            vals = []
            for r in range(br*3, br*3+3):
                for c in range(bc*3, bc*3+3):
                    vals.append(board[r][c])
            if set(vals) != set(range(1, 10)):
                return False
    return True


# ============================================================
# Wrapper Functions for Batch Processing
# ============================================================

def parse_puzzle_string(puzzle_string: str) -> List[List[int]]:
    """Convert 81-char string to 9x9 board."""
    if len(puzzle_string) != 81:
        raise ValueError(f"Puzzle must be 81 characters, got {len(puzzle_string)}")
    
    nums = [int(ch) for ch in puzzle_string]
    board = [nums[i:i+9] for i in range(0, 81, 9)]
    return board


def board_to_string(board: List[List[int]]) -> str:
    """Convert 9x9 board to 81-char string."""
    result = []
    for row in board:
        for val in row:
            result.append(str(val))
    return ''.join(result)


def print_puzzle(puzzle_string: str) -> None:
    """Print puzzle in formatted grid."""
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


def solve_single_puzzle(puzzle_string: str, show_output: bool = True) -> Tuple:
    """
    Solve using Min-Conflicts algorithm.
    Returns: (solution, runtime, memory_mb, nodes, backtracks, success, timed_out)
    
    Note: For Min-Conflicts:
    - nodes = steps/iterations taken
    - backtracks = 0 (no backtracking in local search)
    """
    
    if show_output:
        print("ORIGINAL PUZZLE:")
        print_puzzle(puzzle_string)
    
    # Parse puzzle
    puzzle = parse_puzzle_string(puzzle_string)
    
    timeout_occurred = False
    success = False
    solution_board = None
    steps = 0
    
    # Start tracking
    tracemalloc.start()
    start_time = time.perf_counter()
    
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT_SECONDS)
        
        if show_output:
            print("\nSolving with Min-Conflicts Local Search...")
        
        # Solve using ORIGINAL min_conflicts_solver
        solution_board, steps = min_conflicts_solver(puzzle, MAX_STEPS)
        
        signal.alarm(0)
        
        # Validate solution
        success = is_valid_solution(solution_board)
        
        if success and show_output:
            solution_str = board_to_string(solution_board)
            print("\nSOLVED PUZZLE:")
            print_puzzle(solution_str)
        elif not success and show_output:
            print("\n⚠️  Max iterations reached without valid solution")
        
    except TimeoutException:
        timeout_occurred = True
        success = False
        if show_output:
            print(f"\n⏱️  TIMEOUT: Exceeded {TIMEOUT_SECONDS} seconds")
    except Exception as e:
        if show_output:
            print(f"\n⚠️  ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
        success = False
    finally:
        try:
            signal.alarm(0)
        except:
            pass
    
    # Stop tracking
    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    runtime = end_time - start_time
    peak_mb = peak / (1024 * 1024)
    
    # Get solution string if successful
    solution = board_to_string(solution_board) if success and solution_board else None
    
    # For Min-Conflicts: nodes = steps, backtracks = 0 (no backtracking)
    return solution, runtime, peak_mb, steps, 0, success, timeout_occurred


def read_puzzles_from_file(filename: str) -> List[str]:
    """Read puzzles from text file."""
    puzzles = []
    
    if not os.path.isfile(filename):
        print(f"Warning: File not found: {filename}")
        return puzzles
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        digits_only = ''.join(ch for ch in content if ch.isdigit())
        
        if len(digits_only) % 81 != 0:
            print(f"Warning: {filename} has {len(digits_only)} digits, not a multiple of 81")
        
        for i in range(0, len(digits_only), 81):
            if i + 81 <= len(digits_only):
                puzzles.append(digits_only[i:i+81])
    
    return puzzles


def log_to_csv(row_dict: Dict, csv_filename: str = "performance_log_min_conflict.csv") -> None:
    """Log results to CSV."""
    file_exists = os.path.isfile(csv_filename)
    
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["File", "PuzzleIndex", "GlobalIndex", "Runtime(s)", 
                     "Memory(MB)", "NodesVisited", "Backtracks", "Success", "TimedOut"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row_dict)
        csvfile.flush()


def solve_all_puzzles(filenames: List[str]) -> None:
    """Main function to solve all puzzles."""
    print(f"\n{'='*70}")
    print("SUDOKU SOLVER - Minimum Conflict Local Search")
    print("Algorithm: Min-Conflicts with Row-Swap Repair")
    print(f"Timeout: {TIMEOUT_SECONDS} seconds per puzzle")
    print(f"Max Steps: {MAX_STEPS} per puzzle")
    print(f"{'='*70}\n")
    
    log_file = "performance_log_min_conflict.csv"
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
            print(f"No puzzles found in {filename}")
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
                    print(f"\n✓ Solved in {runtime:.6f}s")
                    print(f"  Steps: {nodes}, Memory: {memory_mb:.6f} MB")
                elif timed_out:
                    print(f"\n⏱️  Timed out after {runtime:.6f}s")
                    print(f"  Steps attempted: {nodes}")
                else:
                    print(f"\n✗ Failed after {runtime:.6f}s")
                    print(f"  Steps attempted: {nodes} (max: {MAX_STEPS})")
                
            except Exception as e:
                print(f"\n✗ ERROR: {str(e)}")
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
    
    # Print summary
    print(f"\n\n{'='*70}")
    print("SUMMARY REPORT")
    print(f"{'='*70}\n")
    
    if all_results:
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
                              "Memory(MB)", "Steps", "Backtracks", "Status"],
                      tablefmt="grid"))
        
        # Statistics by file
        print(f"\n{'='*70}")
        print("STATISTICS BY DIFFICULTY (with Accuracy)")
        print(f"{'='*70}\n")
        
        files = {}
        for r in all_results:
            fname = r["File"]
            if fname not in files:
                files[fname] = {
                    "solved": 0, "total": 0, "timed_out": 0,
                    "total_runtime": 0.0, "total_nodes": 0
                }
            
            files[fname]["total"] += 1
            if r["Success"]:
                files[fname]["solved"] += 1
                files[fname]["total_runtime"] += float(r["Runtime(s)"])
                files[fname]["total_nodes"] += int(r["NodesVisited"])
            if r["TimedOut"]:
                files[fname]["timed_out"] += 1
        
        stats_table = []
        for fname, stats in files.items():
            solved = stats["solved"]
            total = stats["total"]
            timed_out = stats["timed_out"]
            accuracy = (solved / total * 100) if total > 0 else 0
            
            if solved > 0:
                avg_runtime = stats["total_runtime"] / solved
                avg_nodes = stats["total_nodes"] / solved
            else:
                avg_runtime = avg_nodes = 0
            
            stats_table.append([
                fname, f"{solved}/{total}", f"{accuracy:.1f}%", timed_out,
                f"{avg_runtime:.6f}", f"{avg_nodes:.0f}"
            ])
        
        print(tabulate(stats_table,
                      headers=["File", "Solved", "Accuracy", "TimedOut", 
                              "Avg Runtime(s)", "Avg Steps"],
                      tablefmt="grid"))
        
        # Overall statistics
        print(f"\n{'='*70}")
        print("OVERALL STATISTICS")
        print(f"{'='*70}\n")
        
        total_puzzles = len(all_results)
        total_solved = sum(1 for r in all_results if r["Success"])
        total_timed_out = sum(1 for r in all_results if r["TimedOut"])
        overall_accuracy = (total_solved / total_puzzles * 100) if total_puzzles > 0 else 0
        
        overall_stats = [
            ["Total Puzzles", total_puzzles],
            ["Solved", total_solved],
            ["Failed", total_puzzles - total_solved - total_timed_out],
            ["Timed Out", total_timed_out],
            ["Overall Accuracy", f"{overall_accuracy:.2f}%"]
        ]
        
        print(tabulate(overall_stats, tablefmt="simple"))
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {log_file}")
    print(f"Algorithm: Minimum Conflict Local Search")
    print(f"Note: 'NodesVisited' = iterations/steps, 'Backtracks' always 0")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 sudoku_runner_min_conflict.py <file1.txt> [file2.txt] ...")
        print("\nExample: python3 sudoku_runner_min_conflict.py easy.txt medium.txt hard.txt")
        print("\nEach file should contain 81-digit puzzles (use 0 for blanks)")
        sys.exit(1)
    
    input_files = sys.argv[1:]
    solve_all_puzzles(input_files)