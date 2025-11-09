# Modified from: Sawyer Bailey Paccione, Tufts University, 2020
# Original repository: https://github.com/paccionesawyer/sudokuSolver-CSP
# License: MIT
# Modification: simplified version combining Backtracking + Forward Checking only
# Added: add the metrics and able to save it into csv

from SudokuBoard import SudokuBoard
import time
import argparse
import os
import csv
import tracemalloc


###############################################################################
## Domain Setup and Maintenance
###############################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", help="Path to the Sudoku text file", type=str, required=False)
    args = parser.parse_args()
    if args.board:
        return args.board
    else:
        return "testing.txt"  # default file if no argument is passed
 

def set_domains(sudoku_puzzle):
    """Initialize possible domain values for each empty cell."""
    n = 9
    domains = [[[1,2,3,4,5,6,7,8,9] for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            val = sudoku_puzzle.board[i][j]
            if val != 0:
                domains[i][j] = [-1]  # filled cell
                constrict_domains(sudoku_puzzle, i, j, val, domains)
    return domains


def constrict_domains(sudoku_puzzle, row, col, num, domains):
    """Prune num from domains of all cells in the same row, col, and box."""
    constrict_box_domains(sudoku_puzzle, row, col, num, domains)
    constrict_row_domains(sudoku_puzzle, row, col, num, domains)
    constrict_col_domains(sudoku_puzzle, row, col, num, domains)


def constrict_box_domains(sudoku_puzzle, row, col, num, domains):
    box_x = col // 3
    box_y = row // 3
    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if num in domains[i][j] and sudoku_puzzle.board[i][j] == 0:
                domains[i][j].remove(num)


def constrict_row_domains(sudoku_puzzle, row, col, num, domains):
    for j in range(9):
        if num in domains[row][j] and sudoku_puzzle.board[row][j] == 0:
            domains[row][j].remove(num)


def constrict_col_domains(sudoku_puzzle, row, col, num, domains):
    for i in range(9):
        if num in domains[i][col] and sudoku_puzzle.board[i][col] == 0:
            domains[i][col].remove(num)


def repair_domains(sudoku_puzzle, row, col, num, domains):
    """Restore num back to domains after backtracking."""
    box_x, box_y = col // 3, row // 3
    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if sudoku_puzzle.valid_input(i, j, num) and num not in domains[i][j]:
                domains[i][j].append(num)
    for j in range(9):
        if sudoku_puzzle.valid_input(row, j, num) and num not in domains[row][j]:
            domains[row][j].append(num)
    for i in range(9):
        if sudoku_puzzle.valid_input(i, col, num) and num not in domains[i][col]:
            domains[i][col].append(num)


def empty_domain(sudoku_puzzle, domains):
    """Return True if any cell has no valid domain values left."""
    for i in range(9):
        for j in range(9):
            if sudoku_puzzle.board[i][j] == 0 and len(domains[i][j]) == 0:
                return True
    return False

###############################################################################
## Variable Ordering (Heuristic)
###############################################################################

def find_empty_basic(sudoku_puzzle, domains):
    """Find first empty cell (top-left search)."""
    for i in range(9):
        for j in range(9):
            if sudoku_puzzle.board[i][j] == 0:
                return [i, j]
    return False


def minimum_remaining_values(sudoku_puzzle, domains):
    """MRV heuristic — choose cell with fewest remaining domain values."""
    mrv = None
    min_len = 10
    for i in range(9):
        for j in range(9):
            if sudoku_puzzle.board[i][j] == 0 and len(domains[i][j]) < min_len:
                mrv = [i, j]
                min_len = len(domains[i][j])
    return mrv

###############################################################################
## Backtracking + Forward Checking
###############################################################################

def forward_checking(sudoku_puzzle, heuristic):
    """Entry function for backtracking + forward checking."""
    domains = set_domains(sudoku_puzzle)
    sudoku_puzzle.unique_states = 0
    start = time.time()
    solved = forward_checking_rec(sudoku_puzzle, heuristic, domains)
    end = time.time()
    print(f"Solved: {solved}, Time: {round(end-start,3)}s, Steps: {sudoku_puzzle.unique_states}")
    return solved


def forward_checking_rec(sudoku_puzzle, heuristic, domains):
    """Recursive backtracking search with forward checking."""
    next = heuristic(sudoku_puzzle, domains)
    if not next:
        return True

    row, col = next
    sudoku_puzzle.unique_states += 1

    for val in list(domains[row][col]):
        if sudoku_puzzle.valid_input(row, col, val):
            sudoku_puzzle.board[row][col] = val
            constrict_domains(sudoku_puzzle, row, col, val, domains)

            if not empty_domain(sudoku_puzzle, domains):
                if forward_checking_rec(sudoku_puzzle, heuristic, domains):
                    return True

            sudoku_puzzle.board[row][col] = 0
            sudoku_puzzle.backtracks += 1
            repair_domains(sudoku_puzzle, row, col, val, domains)
    return False


# =============================================================
# NEW: CSV Logging Function
# =============================================================
def log_to_csv(row_dict, csv_filename="performance_log_fc.csv"):
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


# =============================================================
# Example Run with CSV logging
# =============================================================
if __name__ == "__main__":

    board_path = parse_arguments()
    puzzle = SudokuBoard(open(board_path, "r"))

    # Measure performance
    start = time.time()
    tracemalloc.start()

    success = forward_checking(puzzle, minimum_remaining_values)

    end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    runtime = end - start
    memory_mb = peak / (1024 * 1024)

    # Print solved board
    puzzle.print_board()

    # Prepare and log results
    row = {
        "File": os.path.basename(board_path),
        "PuzzleIndex": 1,
        "GlobalIndex": 1,
        "Runtime(s)": f"{runtime:.6f}",
        "Memory(MB)": f"{memory_mb:.6f}",
        "NodesVisited": puzzle.unique_states,
        "Backtracks": puzzle.backtracks,
        "Success": success,
        "TimedOut": False
    }

    log_to_csv(row)
    print("Results saved to performance_log_fc.csv")
