#!/usr/bin/env python3
"""
sudoku_runner.py - Fixed Backtracking Algorithm with Timeout and Accuracy

Usage:
  python sudoku_runner.py easy.txt medium.txt hard.txt

This is the CORRECTED version of your original backtracking code.
Each text file should contain 81-digit puzzles (use 0 for blanks).
"""

import sys
import os
import time
import csv
import tracemalloc
import signal
from tabulate import tabulate

# Global variables
ROWS = "ABCDEFGHI"
COLS = "123456789"
DIGITS = "123456789"
squares = []
peers = {}
values = {}

# Timeout settings
TIMEOUT_SECONDS = 30  # Maximum time per puzzle


class TimeoutException(Exception):
    """Custom exception for timeout."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Puzzle solving timed out")


def sudosetup():
    """Initialize the Sudoku grid structure and peers."""
    global squares, peers
    squares = [r + c for r in ROWS for c in COLS]
    
    # Build peers dictionary
    peers = {}
    for square in squares:
        row = square[0]
        col = square[1]
        peer_set = set()
        
        # Row peers
        for c in COLS:
            peer = row + c
            if peer != square:
                peer_set.add(peer)
        
        # Column peers
        for r in ROWS:
            peer = r + col
            if peer != square:
                peer_set.add(peer)
        
        # Box peers
        row_idx = ROWS.index(row)
        col_idx = COLS.index(col)
        box_row_start = (row_idx // 3) * 3
        box_col_start = (col_idx // 3) * 3
        for r in range(box_row_start, box_row_start + 3):
            for c in range(box_col_start, box_col_start + 3):
                peer = ROWS[r] + COLS[c]
                if peer != square:
                    peer_set.add(peer)
        
        peers[square] = peer_set


def sudoprint_formatted(values_dict):
    """Print the Sudoku grid in a formatted style."""
    print()
    for r in range(9):
        row_cells = []
        for c in range(9):
            s = ROWS[r] + COLS[c]
            val = values_dict[s] if values_dict[s] != '0' else '.'
            row_cells.append(val)
        print(" ".join(row_cells[0:3]) + " | " + 
              " ".join(row_cells[3:6]) + " | " + 
              " ".join(row_cells[6:9]))
        if r in [2, 5]:
            print("-" * 21)
    print()


def make_solver():
    """
    Returns solve_sudoku function and counters for metrics.
    This creates a closure to track nodes and backtracks.
    """
    counters = {"nodes": 0, "backtracks": 0}
    
    def is_valid(values, square, digit):
        """Check if placing digit in square is valid."""
        for peer in peers[square]:
            if values[peer] == digit:
                return False
        return True
    
    def find_empty(values):
        """Find the first empty square (containing '0')."""
        for s in squares:
            if values[s] == '0':
                return s
        return None
    
    def solve_sudoku(values):
        """
        Recursive backtracking solver.
        FIXED VERSION: Properly explores all possibilities.
        """
        # Find empty square
        pos = find_empty(values)
        if pos is None:
            return True  # Puzzle solved!
        
        # Try each digit 1-9
        for digit in DIGITS:
            counters["nodes"] += 1
            
            if is_valid(values, pos, digit):
                # Place the digit
                values[pos] = digit
                
                # Recursively solve
                if solve_sudoku(values):
                    return True
                
                # If that didn't work, undo and try next digit
                values[pos] = '0'
                counters["backtracks"] += 1
        
        # No digit worked, backtrack
        return False
    
    return solve_sudoku, counters


def sudosolve_single(puzzle_string, show_output=True):
    """
    Solve a single 81-character Sudoku puzzle with timeout.
    
    Returns:
        (solved_string or None, runtime_s, memory_mb, nodes, backtracks, success, timeout_occurred)
    """
    global values
    
    if len(puzzle_string) != 81:
        raise ValueError(f"Puzzle must be exactly 81 characters, got {len(puzzle_string)}")
    
    # Initialize values dictionary
    values = {}
    for i, char in enumerate(puzzle_string):
        values[squares[i]] = char
    
    if show_output:
        print("ORIGINAL PUZZLE:")
        sudoprint_formatted(values)
    
    # Create solver with counters
    solve_sudoku, counters = make_solver()
    
    # Start tracking
    tracemalloc.start()
    start_time = time.perf_counter()
    
    # Set up timeout alarm (Unix/Mac only)
    timeout_occurred = False
    success = False
    
    try:
        # Set timeout signal
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT_SECONDS)
        
        # SOLVE
        success = solve_sudoku(values)
        
        # Cancel the alarm
        signal.alarm(0)
        
    except TimeoutException:
        timeout_occurred = True
        success = False
        if show_output:
            print(f"\n⏱️  TIMEOUT: Exceeded {TIMEOUT_SECONDS} seconds")
    except Exception as e:
        # For Windows or other errors, handle gracefully
        if show_output:
            print(f"\n⚠️  ERROR: {str(e)}")
        success = False
    finally:
        # Always cancel alarm to be safe
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
    nodes = counters["nodes"]
    backtracks = counters["backtracks"]
    
    if not success:
        if show_output and not timeout_occurred:
            print("✗ FAILED: Puzzle could not be solved!")
        return None, runtime, peak_mb, nodes, backtracks, False, timeout_occurred
    else:
        solved_string = "".join([values[s] for s in squares])
        if show_output:
            print("SOLVED PUZZLE:")
            sudoprint_formatted(values)
        return solved_string, runtime, peak_mb, nodes, backtracks, True, False


def read_puzzles_from_file(filename):
    """
    Read puzzles from a text file.
    Handles both line-by-line puzzles and concatenated strings.
    
    Returns:
        List of 81-character puzzle strings
    """
    puzzles = []
    
    if not os.path.isfile(filename):
        print(f"Warning: File not found: {filename}")
        return puzzles
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # Remove all non-digit characters
        digits_only = ''.join(ch for ch in content if ch.isdigit())
        
        # Split into 81-character chunks
        if len(digits_only) % 81 != 0:
            print(f"Warning: {filename} has {len(digits_only)} digits, not a multiple of 81")
        
        for i in range(0, len(digits_only), 81):
            if i + 81 <= len(digits_only):
                puzzles.append(digits_only[i:i+81])
    
    return puzzles


def log_to_csv(row_dict, csv_filename="performance_log.csv"):
    """Append a result row to the CSV log file."""
    file_exists = os.path.isfile(csv_filename)
    
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["File", "PuzzleIndex", "GlobalIndex", "Runtime(s)", 
                     "Memory(MB)", "NodesVisited", "Backtracks", "Success", "TimedOut"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row_dict)
        csvfile.flush()


def solve_all_puzzles(filenames):
    """
    Main function to solve all puzzles from the provided files.
    """
    print(f"\n{'='*70}")
    print("SUDOKU SOLVER - Basic Backtracking Algorithm")
    print(f"Timeout: {TIMEOUT_SECONDS} seconds per puzzle")
    print(f"{'='*70}\n")
    
    # Clear existing log
    log_file = "performance_log.csv"
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
                    sudosolve_single(puzzle, show_output=True)
                
                # Prepare result row
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
                
                # Log to CSV
                log_to_csv(row, log_file)
                all_results.append(row)
                
                # Print summary
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
                
                # Log error
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
    
    # Print summary table
    print(f"\n\n{'='*70}")
    print("SUMMARY REPORT")
    print(f"{'='*70}\n")
    
    if all_results:
        table_data = []
        for r in all_results:
            status = "✓" if r["Success"] else ("⏱️" if r["TimedOut"] else "✗")
            table_data.append([
                r["GlobalIndex"],
                r["File"],
                r["PuzzleIndex"],
                r["Runtime(s)"],
                r["Memory(MB)"],
                r["NodesVisited"],
                r["Backtracks"],
                status
            ])
        
        print(tabulate(table_data, 
                      headers=["Global#", "File", "Puzzle#", "Runtime(s)", 
                              "Memory(MB)", "Nodes", "Backtracks", "Status"],
                      tablefmt="grid"))
        
        # Statistics by file with ACCURACY
        print(f"\n{'='*70}")
        print("STATISTICS BY DIFFICULTY (with Accuracy)")
        print(f"{'='*70}\n")
        
        files = {}
        for r in all_results:
            fname = r["File"]
            if fname not in files:
                files[fname] = {
                    "solved": 0, 
                    "total": 0, 
                    "timed_out": 0,
                    "total_runtime": 0.0, 
                    "total_nodes": 0, 
                    "total_backtracks": 0
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
            
            # ACCURACY = (Solved / Total) * 100%
            accuracy = (solved / total * 100) if total > 0 else 0
            
            if solved > 0:
                avg_runtime = stats["total_runtime"] / solved
                avg_nodes = stats["total_nodes"] / solved
                avg_backtracks = stats["total_backtracks"] / solved
            else:
                avg_runtime = 0
                avg_nodes = 0
                avg_backtracks = 0
            
            stats_table.append([
                fname,
                f"{solved}/{total}",
                f"{accuracy:.1f}%",
                timed_out,
                f"{avg_runtime:.6f}",
                f"{avg_nodes:.0f}",
                f"{avg_backtracks:.0f}"
            ])
        
        print(tabulate(stats_table,
                      headers=["File", "Solved", "Accuracy", "TimedOut", 
                              "Avg Runtime(s)", "Avg Nodes", "Avg Backtracks"],
                      tablefmt="grid"))
        
        # Overall accuracy
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
    print(f"Algorithm: Basic Backtracking")
    print(f"Timeout per puzzle: {TIMEOUT_SECONDS} seconds")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sudoku_runner.py <file1.txt> [file2.txt] ...")
        print("\nExample: python sudoku_runner.py easy.txt medium.txt hard.txt")
        print("\nEach file should contain 81-digit puzzles (use 0 for blanks)")
        sys.exit(1)
    
    # Initialize Sudoku setup
    sudosetup()
    
    # Get input files from command line
    input_files = sys.argv[1:]
    
    # Solve all puzzles
    solve_all_puzzles(input_files)