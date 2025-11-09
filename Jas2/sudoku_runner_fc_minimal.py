#!/usr/bin/env python3
"""
sudoku_runner_fc_minimal.py

Wrapper for Backtracking + Forward Checking + MRV implementation
Source: https://github.com/paccionesawyer/sudokuSolver-CSP

Algorithm: Backtracking + Forward Checking + MRV Heuristic

MINIMAL CHANGES:
1. Read from easy.txt, medium.txt, hard.txt
2. Track metrics (already built-in!)
3. Output to CSV in your format

ORIGINAL CODE UNCHANGED - Just wrapped for batch processing
"""

import sys
import os
import time
import csv
import tracemalloc
import signal
from tabulate import tabulate
from typing import List, Dict, Tuple
from io import StringIO

# Import original modules
from SudokuBoard import SudokuBoard
from solver import forward_checking, minimum_remaining_values

TIMEOUT_SECONDS = 30


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Timeout")


def solve_single_puzzle(puzzle_string: str, show_output: bool = True) -> Tuple:
    """
    Solve using ORIGINAL Forward Checking implementation.
    Returns: (solution, runtime, memory_mb, nodes, backtracks, success, timed_out)
    """
    
    if len(puzzle_string) != 81:
        raise ValueError(f"Puzzle must be 81 characters, got {len(puzzle_string)}")
    
    # Print original puzzle
    if show_output:
        print("ORIGINAL PUZZLE:")
        print_puzzle(puzzle_string)
    
    timeout_occurred = False
    success = False
    solution = None
    
    # Start tracking
    tracemalloc.start()
    start_time = time.perf_counter()
    
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT_SECONDS)
        
        # Create puzzle from string (using StringIO to simulate file)
        file_obj = StringIO(puzzle_string)
        puzzle = SudokuBoard(file_obj)
        
        if show_output:
            print("\nSolving with Backtracking + Forward Checking + MRV...")
        
        # Solve using ORIGINAL forward_checking function
        success = forward_checking(puzzle, minimum_remaining_values)
        
        signal.alarm(0)
        
        # Get solution if successful
        if success:
            solution = board_to_string(puzzle)
            if show_output:
                print("\nSOLVED PUZZLE:")
                puzzle.print_board()
        
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
    
    # Get metrics from puzzle object (already tracked by original code!)
    nodes = puzzle.unique_states if hasattr(puzzle, 'unique_states') else 0
    backtracks = puzzle.backtracks if hasattr(puzzle, 'backtracks') else 0
    
    return solution, runtime, peak_mb, nodes, backtracks, success, timeout_occurred


def board_to_string(puzzle) -> str:
    """Convert SudokuBoard back to 81-character string"""
    result = []
    for i in range(9):
        for j in range(9):
            result.append(str(puzzle.board[i][j]))
    return ''.join(result)


def print_puzzle(puzzle_string: str) -> None:
    """Print puzzle in formatted grid"""
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


def log_to_csv(row_dict: Dict, csv_filename: str = "performance_log_fc_minimal.csv") -> None:
    """Log results to CSV"""
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
    """Main function"""
    print(f"\n{'='*70}")
    print("SUDOKU SOLVER - Backtracking + Forward Checking + MRV")
    print("Source: github.com/paccionesawyer/sudokuSolver-CSP")
    print(f"Timeout: {TIMEOUT_SECONDS} seconds per puzzle")
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
                    print(f"  Nodes: {nodes}, Backtracks: {backtracks}, Memory: {memory_mb:.6f} MB")
                elif timed_out:
                    print(f"\n⏱️  Timed out after {runtime:.6f}s")
                    print(f"  Nodes: {nodes}, Backtracks: {backtracks}")
                else:
                    print(f"\n✗ Failed after {runtime:.6f}s")
                    print(f"  Nodes: {nodes}, Backtracks: {backtracks}")
                
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
                              "Memory(MB)", "Nodes", "Backtracks", "Status"],
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
                    "total_runtime": 0.0, "total_nodes": 0, "total_backtracks": 0
                }
            
            files[fname]["total"] += 1
            if r["Success"]:
                files[fname]["solved"] += 1
                files[fname]["total_runtime"] += float(r["Runtime(s)"])
                files[fname]["total_nodes"] += int(r["NodesVisited"])
                files[fname]["total_backtracks"] += int(r["Backtracks"])
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
                avg_backtracks = stats["total_backtracks"] / solved
            else:
                avg_runtime = avg_nodes = avg_backtracks = 0
            
            stats_table.append([
                fname, f"{solved}/{total}", f"{accuracy:.1f}%", timed_out,
                f"{avg_runtime:.6f}", f"{avg_nodes:.0f}", f"{avg_backtracks:.0f}"
            ])
        
        print(tabulate(stats_table,
                      headers=["File", "Solved", "Accuracy", "TimedOut", 
                              "Avg Runtime(s)", "Avg Nodes", "Avg Backtracks"],
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
    print(f"Algorithm: Backtracking + Forward Checking + MRV")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 sudoku_runner_fc_minimal.py <file1.txt> [file2.txt] ...")
        print("\nExample: python3 sudoku_runner_fc_minimal.py easy.txt medium.txt hard.txt")
        print("\nEach file should contain 81-digit puzzles (use 0 for blanks)")
        sys.exit(1)
    
    input_files = sys.argv[1:]
    solve_all_puzzles(input_files)