import sys
import os
import time
import csv
import tracemalloc
from tabulate import tabulate
from typing import List, Dict

# ===============================================================
# Original Sudoku setup
# ===============================================================
ROWS = "ABCDEFGHI"
COLS = "123456789"
DIGITS = COLS
squares = [r + c for r in ROWS for c in COLS]
values = {}

# Precomputed peers, lines, and units
def sudosetup():
    global units, unit, lines, peers
    unitrows = []
    for r in range(3):
        for i in range(3):
            unitrows.append(list(map(lambda x: x + r * 3,
                                     [1, 1, 1, 2, 2, 2, 3, 3, 3])))
    unitlist = [j for i in unitrows for j in i]

    units = {i: [] for i in range(1, 10)}
    for (i, s) in enumerate(squares):
        units[unitlist[i]].append(s)

    unit = {squares[i]: unitlist[i] for i in range(81)}
    lines = {}
    for row_or_col in ROWS + COLS:
        members = []
        for s in squares:
            if row_or_col in s:
                members.append(s)
        lines[row_or_col] = members

    peers = {}
    for s in squares:
        peerlist = [units[unit[s]], lines[s[0]] + lines[s[1]]]
        peers[s] = set(p for p in [j for i in peerlist for j in i] if p != s)


# ===============================================================
# Sudoku validation & brute-force solving
# ===============================================================
def sudo_validate():
    if "0" in values.values():
        return False
    for l in lines.keys():
        if set(values[s] for s in lines[l]) != set(DIGITS):
            return False
    for u in units.keys():
        if set(values[s] for s in units[u]) != set(DIGITS):
            return False
    return True


def sudo_brute_force(counters):
    """Recursive brute-force solver with node/backtrack counters."""
    if sudo_validate():
        return True

    # Find first empty cell
    current_square = None
    for s, v in values.items():
        if v == "0":
            current_square = s
            break

    cs_peer_values = [values[s] for s in peers[current_square]]
    cs_possibilities = [d for d in DIGITS if d not in cs_peer_values]
    if not cs_possibilities:
        counters["backtracks"] += 1
        return False

    for d in cs_possibilities:
        counters["nodes"] += 1
        values[current_square] = d
        if sudo_brute_force(counters):
            return True

    values[current_square] = "0"
    counters["backtracks"] += 1
    return False


# ===============================================================
# Single puzzle solver
# ===============================================================
def solve_single_puzzle(puzzle_string: str, show_output=True):
    global values
    if len(puzzle_string) != 81:
        raise ValueError(
            f"Puzzle must be exactly 81 characters, got {len(puzzle_string)}"
        )

    # Assign puzzle values
    values = {squares[i]: puzzle_string[i] for i in range(81)}

    if show_output:
        print("ORIGINAL PUZZLE:")
        print_puzzle(values)

    counters = {"nodes": 0, "backtracks": 0}

    tracemalloc.start()
    start = time.perf_counter()

    success = False
    try:
        success = sudo_brute_force(counters)
    except Exception as e:
        print(f" ERROR: {e}")
        success = False

    end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    runtime = end - start
    memory_mb = peak / (1024 * 1024)

    solved_string = "".join(values[s] for s in squares)
    if show_output and success:
        print("SOLVED PUZZLE:")
        print_puzzle(values)

    # No TimedOut flag anymore
    return (
        solved_string if success else None,
        runtime,
        memory_mb,
        counters["nodes"],
        counters["backtracks"],
        success,
    )


# ===============================================================
# Print Sudoku nicely
# ===============================================================
def print_puzzle(values):
    print()
    for r in range(9):
        row = [values[ROWS[r] + COLS[c] ] for c in range(9)]
        print(
            " ".join(row[0:3]) + " | "
            + " ".join(row[3:6]) + " | "
            + " ".join(row[6:9])
        )
        if r in [2, 5]:
            print("-" * 21)
    print()


# ===============================================================
# CSV + batch solving
# ===============================================================
def read_puzzles_from_file(filename):
    puzzles = []
    if not os.path.isfile(filename):
        print(f" File not found: {filename}")
        return puzzles
    with open(filename, "r", encoding="utf-8") as f:
        digits = "".join(ch for ch in f.read() if ch.isdigit())
        for i in range(0, len(digits), 81):
            if i + 81 <= len(digits):
                puzzles.append(digits[i:i+81])
    return puzzles


def log_to_csv(row: Dict, csv_filename="performance_log_bruteforce.csv"):
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, "a", newline="", encoding="utf-8") as f:
        fieldnames = [
            "File", "PuzzleIndex", "GlobalIndex", "Runtime(s)",
            "Memory(MB)", "NodesVisited", "Backtracks", "Success"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def solve_all_puzzles(filenames: List[str]):
    print(f"\n{'='*70}")
    print("SUDOKU SOLVER - Brute Force Backtracking")
    print("Timeout: DISABLED (no time limit)")
    print(f"{'='*70}\n")

    log_file = "performance_log_bruteforce.csv"
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
            solved, runtime, mem, nodes, backs, success = solve_single_puzzle(
                puzzle, show_output=True
            )

            row = {
                "File": os.path.basename(filename),
                "PuzzleIndex": idx,
                "GlobalIndex": global_index,
                "Runtime(s)": f"{runtime:.6f}",
                "Memory(MB)": f"{mem:.6f}",
                "NodesVisited": nodes,
                "Backtracks": backs,
                "Success": success,
            }
            log_to_csv(row, log_file)
            all_results.append(row)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY REPORT")
    print(f"{'='*70}\n")

    if not all_results:
        print("No puzzles processed.")
        return

    table = []
    for r in all_results:
        status = "✓" if r["Success"] else "✗"
        table.append([
            r["GlobalIndex"], r["File"], r["PuzzleIndex"], r["Runtime(s)"],
            r["Memory(MB)"], r["NodesVisited"], r["Backtracks"], status
        ])
    print(tabulate(
        table,
        headers=[
            "Global#", "File", "Puzzle#", "Runtime(s)", "Memory(MB)",
            "Nodes", "Backtracks", "Status"
        ],
        tablefmt="grid"
    ))

    # Statistics by difficulty (NO TimedOut column)
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

    stats = []
    for fname, s in files.items():
        acc = (s["solved"] / s["total"] * 100) if s["total"] > 0 else 0
        avg_runtime = s["runtime"] / s["solved"] if s["solved"] > 0 else 0
        avg_nodes = s["nodes"] / s["solved"] if s["solved"] > 0 else 0
        avg_back = s["backs"] / s["solved"] if s["solved"] > 0 else 0
        stats.append([
            fname, f"{s['solved']}/{s['total']}", f"{acc:.1f}%",
            f"{avg_runtime:.6f}", f"{avg_nodes:.0f}", f"{avg_back:.0f}"
        ])

    print(tabulate(
        stats,
        headers=["File", "Solved", "Accuracy",
                 "Avg Runtime(s)", "Avg Nodes", "Avg Backtracks"],
        tablefmt="grid"
    ))

    # Overall stats (NO 'Timed Out' row)
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
        ["Overall Accuracy", f"{acc:.2f}%"],
    ]
    print(tabulate(overall, tablefmt="simple"))
    print(f"\n{'='*70}")
    print(f"Results saved to: {log_file}")
    print(f"Algorithm: Brute Force Backtracking (NO TIMEOUT)")
    print(f"{'='*70}\n")


# ===============================================================
# Entry Point
# ===============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sudoku_runner_bruteforce.py easy.txt medium.txt hard.txt")
        sys.exit(1)

    sudosetup()
    solve_all_puzzles(sys.argv[1:])
