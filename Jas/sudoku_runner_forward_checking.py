#!/usr/bin/env python3
"""
sudoku_runner_forward_checking.py
Backtracking + Forward Checking + Heuristics (MRV + LCV)

Usage:
  python3 sudoku_runner_forward_checking.py easy.txt medium.txt hard.txt

This implements the algorithm from the Java code:
- Forward Checking (constraint propagation)
- MRV Heuristic (Minimum Remaining Values)
- LCV Heuristic (Least Constraining Value)
"""

import sys
import os
import time
import csv
import tracemalloc
import signal
from tabulate import tabulate
from typing import Dict, List, Set, Optional, Tuple

# Global variables
ROWS = "ABCDEFGHI"
COLS = "123456789"
DIGITS = "123456789"
TIMEOUT_SECONDS = 30

class TimeoutException(Exception):
    """Custom exception for timeout."""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Puzzle solving timed out")


class SudokuNode:
    """Represents a single cell in the Sudoku grid."""
    
    def __init__(self, row: int, col: int, value: Optional[int] = None):
        self.row = row
        self.col = col
        self.box = self._calc_box(row, col)
        
        if value is not None and 1 <= value <= 9:
            self.domain = {value}
        else:
            self.domain = set(range(1, 10))
    
    def _calc_box(self, row: int, col: int) -> int:
        """Calculate which 3x3 box this cell belongs to (0-8)."""
        return (row // 3) * 3 + (col // 3)
    
    def __repr__(self):
        return f"({self.row},{self.col})"


class ForwardCheckingSolver:
    """Sudoku solver using Forward Checking + MRV + LCV heuristics."""
    
    def __init__(self):
        self.nodes: List[List[SudokuNode]] = []
        self.squares: List[str] = [r + c for r in ROWS for c in COLS]
        self.step_count = 0
        self.backtrack_count = 0
        self.stack: List[Tuple[SudokuNode, int]] = []
    
    def parse_puzzle(self, puzzle_string: str) -> None:
        """Initialize the grid from 81-character puzzle string."""
        if len(puzzle_string) != 81:
            raise ValueError(f"Puzzle must be 81 characters, got {len(puzzle_string)}")
        
        self.nodes = []
        for i in range(9):
            row = []
            for j in range(9):
                idx = i * 9 + j
                char = puzzle_string[idx]
                value = int(char) if char != '0' else None
                row.append(SudokuNode(i, j, value))
            self.nodes.append(row)
    
    def get_neighbors(self, node: SudokuNode) -> List[SudokuNode]:
        """Get all neighbors (same row, column, or box) with domain size > 1."""
        neighbors = []
        
        # Same row
        for j in range(9):
            if j != node.col and len(self.nodes[node.row][j].domain) > 1:
                neighbors.append(self.nodes[node.row][j])
        
        # Same column
        for i in range(9):
            if i != node.row and len(self.nodes[i][node.col].domain) > 1:
                neighbors.append(self.nodes[i][node.col])
        
        # Same box
        for i in range(9):
            for j in range(9):
                if (i != node.row and j != node.col and 
                    self.nodes[i][j].box == node.box and
                    len(self.nodes[i][j].domain) > 1):
                    neighbors.append(self.nodes[i][j])
        
        return neighbors
    
    def forward_check(self) -> bool:
        """
        Establish arc consistency for nodes with single value in domain.
        Returns False if inconsistency found, True otherwise.
        """
        is_continue = False
        
        for i in range(9):
            for j in range(9):
                if len(self.nodes[i][j].domain) == 1:
                    neighbors = self.get_neighbors(self.nodes[i][j])
                    value = list(self.nodes[i][j].domain)[0]
                    
                    for neighbor in neighbors:
                        if not self.remove_node_value(neighbor, value):
                            return False
                        elif len(neighbor.domain) == 1:
                            is_continue = True
        
        if is_continue:
            return self.forward_check()
        
        return True
    
    def get_heuristic_node(self) -> Optional[SudokuNode]:
        """
        MRV Heuristic: Select unassigned node with minimum remaining values.
        Returns None if all nodes are assigned.
        """
        for size in range(2, 10):
            for i in range(9):
                for j in range(9):
                    if len(self.nodes[i][j].domain) == size:
                        return self.nodes[i][j]
        return None
    
    def get_heuristic_value(self, node: SudokuNode) -> int:
        """
        LCV Heuristic: Select value that constrains neighbors least.
        Returns the least constraining value (or 0 if domain is empty).
        """
        if not node.domain:
            return 0
        
        neighbors = self.get_neighbors(node)
        count_map = {}
        
        for value in node.domain:
            count = 0
            for neighbor in neighbors:
                count += len(neighbor.domain)
                if value in neighbor.domain:
                    count -= 1
            count_map[value] = count
        
        # Return value with maximum count (least constraining)
        if not count_map:
            return 0
        
        return max(count_map.keys(), key=lambda k: count_map[k])
    
    def remove_node_value(self, node: SudokuNode, value: int) -> bool:
        """
        Remove a value from node's domain and record in stack.
        Returns False if domain becomes empty, True otherwise.
        """
        if value in node.domain:
            node.domain.remove(value)
            self.stack.append((node, value))
            if not node.domain:
                return False
        return True
    
    def set_node_value(self, node: SudokuNode, value: int) -> None:
        """Assign a value to node by removing all other values."""
        values_to_remove = [v for v in node.domain if v != value]
        for v in values_to_remove:
            node.domain.remove(v)
            self.stack.append((node, v))
    
    def backtrack_state(self, state: int) -> None:
        """Restore domain values to a previous state."""
        while len(self.stack) > state:
            node, value = self.stack.pop()
            node.domain.add(value)
    
    def judge_state(self) -> int:
        """
        Judge the current state of the puzzle.
        Returns: -1 (error), 0 (solved), 1 (not finished)
        """
        not_end = False
        
        for i in range(9):
            for j in range(9):
                domain_size = len(self.nodes[i][j].domain)
                if domain_size == 0:
                    return -1
                elif domain_size > 1:
                    not_end = True
        
        if not_end:
            return 1
        
        # Check validity of solution
        # Check rows and columns
        for i in range(9):
            row_set = set()
            col_set = set()
            for j in range(9):
                row_set.add(list(self.nodes[i][j].domain)[0])
                col_set.add(list(self.nodes[j][i].domain)[0])
            if len(row_set) != 9 or len(col_set) != 9:
                return -1
        
        # Check boxes
        box_sums = {}
        for i in range(9):
            for j in range(9):
                box = self.nodes[i][j].box
                value = list(self.nodes[i][j].domain)[0]
                box_sums[box] = box_sums.get(box, 0) + value
        
        for box_sum in box_sums.values():
            if box_sum != 45:
                return -1
        
        return 0
    
    def search(self, node: Optional[SudokuNode]) -> int:
        """
        Recursive search with forward checking.
        Returns: -1 (error), 0 (solved), 1 (not finished)
        """
        state = len(self.stack)
        
        result = self.judge_state()
        if result != 1:
            return result
        
        if node is None:
            return -1
        
        value = self.get_heuristic_value(node)
        while value != 0:
            self.step_count += 1
            
            self.set_node_value(node, value)
            
            if not self.forward_check():
                # This value is invalid
                self.backtrack_state(state)
                self.backtrack_count += 1
                if not self.remove_node_value(node, value):
                    return -1
                state = len(self.stack)
                value = self.get_heuristic_value(node)
            else:
                # This value is valid, search next node
                result = self.search(self.get_heuristic_node())
                
                if result == 0:
                    return 0
                elif result == -1:
                    self.backtrack_state(state)
                    self.backtrack_count += 1
                    if not self.remove_node_value(node, value):
                        return -1
                    state = len(self.stack)
                
                value = self.get_heuristic_value(node)
        
        return -1
    
    def solve(self) -> bool:
        """
        Main solve method. Returns True if solved, False otherwise.
        """
        # Initial forward checking
        if not self.forward_check():
            return False
        
        result = self.judge_state()
        if result == 0:
            return True
        elif result == -1:
            return False
        
        # Start backtracking search
        result = self.search(self.get_heuristic_node())
        return result == 0
    
    def get_solution_string(self) -> str:
        """Convert the solved grid back to 81-character string."""
        result = []
        for i in range(9):
            for j in range(9):
                if len(self.nodes[i][j].domain) == 1:
                    result.append(str(list(self.nodes[i][j].domain)[0]))
                else:
                    result.append('0')
        return ''.join(result)
    
    def print_grid(self) -> None:
        """Print the grid in formatted style."""
        print()
        for i in range(9):
            row_cells = []
            for j in range(9):
                if len(self.nodes[i][j].domain) == 1:
                    row_cells.append(str(list(self.nodes[i][j].domain)[0]))
                else:
                    row_cells.append('.')
            print(" ".join(row_cells[0:3]) + " | " + 
                  " ".join(row_cells[3:6]) + " | " + 
                  " ".join(row_cells[6:9]))
            if i in [2, 5]:
                print("-" * 21)
        print()


def solve_puzzle_with_timeout(puzzle_string: str, show_output: bool = True) -> Tuple:
    """
    Solve a puzzle with timeout and metrics tracking.
    Returns: (solution_str, runtime, memory_mb, nodes, backtracks, success, timed_out)
    """
    solver = ForwardCheckingSolver()
    solver.parse_puzzle(puzzle_string)
    
    if show_output:
        print("ORIGINAL PUZZLE:")
        solver.print_grid()
    
    # Start tracking
    tracemalloc.start()
    start_time = time.perf_counter()
    
    timeout_occurred = False
    success = False
    
    try:
        # Set timeout signal
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT_SECONDS)
        
        # SOLVE
        success = solver.solve()
        
        # Cancel the alarm
        signal.alarm(0)
        
    except TimeoutException:
        timeout_occurred = True
        success = False
        if show_output:
            print(f"\n⏱️  TIMEOUT: Exceeded {TIMEOUT_SECONDS} seconds")
    except Exception as e:
        if show_output:
            print(f"\n⚠️  ERROR: {str(e)}")
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
    nodes = solver.step_count
    backtracks = solver.backtrack_count
    
    if success:
        solution = solver.get_solution_string()
        if show_output:
            print("SOLVED PUZZLE:")
            solver.print_grid()
        return solution, runtime, peak_mb, nodes, backtracks, True, False
    else:
        if show_output and not timeout_occurred:
            print("✗ FAILED: Puzzle could not be solved!")
        return None, runtime, peak_mb, nodes, backtracks, False, timeout_occurred


def read_puzzles_from_file(filename: str) -> List[str]:
    """Read puzzles from a text file."""
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


def log_to_csv(row_dict: Dict, csv_filename: str = "performance_log_fc.csv") -> None:
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


def solve_all_puzzles(filenames: List[str]) -> None:
    """Main function to solve all puzzles from the provided files."""
    print(f"\n{'='*70}")
    print("SUDOKU SOLVER - Forward Checking + Heuristics (MRV + LCV)")
    print(f"Timeout: {TIMEOUT_SECONDS} seconds per puzzle")
    print(f"{'='*70}\n")
    
    log_file = "performance_log_fc.csv"
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
                    solve_puzzle_with_timeout(puzzle, show_output=True)
                
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
    print(f"Algorithm: Forward Checking + Heuristics (MRV + LCV)")
    print(f"Timeout per puzzle: {TIMEOUT_SECONDS} seconds")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 sudoku_runner_forward_checking.py <file1.txt> [file2.txt] ...")
        print("\nExample: python3 sudoku_runner_forward_checking.py easy.txt medium.txt hard.txt")
        print("\nEach file should contain 81-digit puzzles (use 0 for blanks)")
        sys.exit(1)
    
    input_files = sys.argv[1:]
    solve_all_puzzles(input_files)